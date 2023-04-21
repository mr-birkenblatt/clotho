import logging
import os
from typing import Callable, cast, IO, Iterable, Literal, TypedDict

import torch
from torch import nn

# FIXME: add transformer stubs
from transformers import (  # type: ignore
    DistilBertModel,
    DistilBertTokenizer,
    modeling_utils,
)

from db.db import DBConnector
from misc.io import listdir, open_read, remove_file
from misc.util import (
    extract_number,
    get_file_hash,
    highest_number,
    json_load,
    retain_some,
    safe_ravel,
)
from model.embedding import (
    EmbeddingProvider,
    EmbeddingProviderMap,
    PROVIDER_CHILD,
    PROVIDER_PARENT,
    ProviderRole,
    STORAGE_ARRAY,
    StorageMethod,
)
from system.embedding.dbcache import read_db_model
from system.msgs.message import Message


EMBED_SIZE = 768


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


TokenizedInput = TypedDict('TokenizedInput', {
    "input_ids": torch.Tensor,
    "attention_mask": torch.Tensor,
})


EpochStats = TypedDict('EpochStats', {
    "epoch": int,
    "train_acc": float,
    "train_loss": float,
    "train_val_acc": float,
    "train_val_loss": float,
    "test_acc": float,
    "test_loss": float,
    "time": float,
    "version": int,
    "fname": str,
})


AggType = Literal["cls", "mean"]
AGG_CLS: AggType = "cls"
AGG_MEAN: AggType = "mean"


def get_tokenizer() -> Callable[[list[str]], TokenizedInput]:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    device = get_device()

    def tokens(texts: list[str]) -> TokenizedInput:
        res = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True)
        return cast(TokenizedInput, {k: v.to(device) for k, v in res.items()})

    return tokens


class Noise(nn.Module):
    def __init__(self, std: float = 1.0, p: float = 0.5) -> None:
        super().__init__()
        self._std = std
        self._p = p
        self._dhold = nn.Parameter(torch.Tensor([0.0]), requires_grad=False)

    def set_std(self, std: float) -> None:
        self._std = std

    def get_std(self) -> float:
        return self._std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        prob = torch.rand(size=x.shape, device=self._dhold.device) < self._p
        gauss = torch.normal(
            mean=0.0, std=self._std, size=x.shape, device=self._dhold.device)
        return x + prob * gauss


def batch_dot(batch_a: torch.Tensor, batch_b: torch.Tensor) -> torch.Tensor:
    batch_size = batch_a.shape[0]
    return torch.bmm(
        batch_a.reshape([batch_size, 1, -1]),
        batch_b.reshape([batch_size, -1, 1])).reshape([-1, 1])


def dot_as_distance(dot: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(-dot))


def cos_as_distance(cos: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(max(0.0, 2.0 - 2.0 * cos))


class Model(nn.Module):
    def __init__(
            self,
            version: int,
            *,
            ignore_pretrained_warning: bool = False) -> None:
        super().__init__()
        assert version >= 0
        assert version <= 7
        logger = modeling_utils.logger
        level = logger.getEffectiveLevel()
        try:
            if ignore_pretrained_warning:
                logger.setLevel(logging.ERROR)
            self._bert_parent = DistilBertModel.from_pretrained(
                "distilbert-base-uncased")
            self._bert_child = DistilBertModel.from_pretrained(
                "distilbert-base-uncased")
        finally:
            if ignore_pretrained_warning:
                logger.setLevel(level)
        if version in (1, 3, 4, 6):
            self._pdense: nn.Sequential | None = nn.Sequential(
                nn.Linear(EMBED_SIZE, EMBED_SIZE),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(EMBED_SIZE, EMBED_SIZE))
            self._cdense: nn.Sequential | None = nn.Sequential(
                nn.Linear(EMBED_SIZE, EMBED_SIZE),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(EMBED_SIZE, EMBED_SIZE))
        else:
            self._pdense = None
            self._cdense = None
        if version < 4 or version > 5:
            self._noise = None
        else:
            self._noise = Noise(std=1.0, p=0.2)
        if version < 2 or version > 4:
            self._cos = None
        else:
            self._cos = torch.nn.CosineSimilarity()
        if version < 6:
            self._agg = AGG_CLS
        else:
            self._agg = AGG_MEAN
        self._version = version

    def set_epoch(self, epoch: int) -> None:
        noise = self._noise
        if noise is not None:
            noise.set_std(1 / (1.2 ** epoch))

    def get_version(self) -> int:
        return self._version

    def get_agg(self, lhs: torch.Tensor) -> torch.Tensor:
        if self._agg == AGG_CLS:
            return lhs[:, 0]
        if self._agg == AGG_MEAN:
            return torch.mean(lhs, dim=1)
        raise ValueError(f"unknown aggregation: {self._agg}")

    def get_parent_embed(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor) -> torch.Tensor:
        outputs_parent = self._bert_parent(
            input_ids=input_ids, attention_mask=attention_mask)
        out = self.get_agg(outputs_parent.last_hidden_state)
        if self._pdense is not None:
            out = self._pdense(out)
        if self._noise is not None:
            out = self._noise(out)
        return out

    def get_child_embed(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor) -> torch.Tensor:
        outputs_child = self._bert_child(
            input_ids=input_ids, attention_mask=attention_mask)
        out = self.get_agg(outputs_child.last_hidden_state)
        if self._cdense is not None:
            out = self._cdense(out)
        if self._noise is not None:
            out = self._noise(out)
        return out

    def forward(
            self,
            x: dict[ProviderRole, TokenizedInput],
            *,
            as_distance: bool = False) -> torch.Tensor:
        parent_embed = self.get_parent_embed(
            input_ids=x["parent"]["input_ids"],
            attention_mask=x["parent"]["attention_mask"])
        child_embed = self.get_child_embed(
            input_ids=x["child"]["input_ids"],
            attention_mask=x["child"]["attention_mask"])
        if self._cos is not None:
            res = self._cos(parent_embed, child_embed).reshape([-1, 1])
            return cos_as_distance(res) if as_distance else res
        res = batch_dot(parent_embed, child_embed)
        return dot_as_distance(res) if as_distance else res


class BaselineModel(nn.Module):
    def __init__(
            self,
            version: int,
            *,
            ignore_pretrained_warning: bool = False) -> None:
        super().__init__()
        assert version < 0
        assert version >= -3
        logger = modeling_utils.logger
        level = logger.getEffectiveLevel()
        try:
            if ignore_pretrained_warning:
                logger.setLevel(logging.ERROR)
            self._bert = DistilBertModel.from_pretrained(
                "distilbert-base-uncased")
        finally:
            if ignore_pretrained_warning:
                logger.setLevel(level)
        if version == -2:
            self._agg = AGG_CLS
        else:
            self._agg = AGG_MEAN
        if version != -3:
            self._dense = None
        else:
            self._dense = nn.Sequential(
                nn.Linear(EMBED_SIZE, EMBED_SIZE),
                nn.Dropout(p=0.2),
                nn.ReLU(),
                nn.Linear(EMBED_SIZE, EMBED_SIZE))
        self._version = version

    def set_epoch(self, epoch: int) -> None:
        pass

    def get_version(self) -> int:
        return self._version

    def get_agg(self, lhs: torch.Tensor) -> torch.Tensor:
        if self._agg == AGG_CLS:
            return lhs[:, 0]
        if self._agg == AGG_MEAN:
            return torch.mean(lhs, dim=1)
        raise ValueError(f"unknown aggregation: {self._agg}")

    def _embed(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            *,
            use_dense: bool) -> torch.Tensor:
        outputs = self._bert(
            input_ids=input_ids, attention_mask=attention_mask)
        out = self.get_agg(outputs.last_hidden_state)
        if use_dense and self._dense is not None:
            out = self._dense(out)
        return out

    def get_parent_embed(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor) -> torch.Tensor:
        return self._embed(input_ids, attention_mask, use_dense=True)

    def get_child_embed(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor) -> torch.Tensor:
        return self._embed(input_ids, attention_mask, use_dense=False)

    def forward(
            self,
            x: dict[ProviderRole, TokenizedInput],
            *,
            as_distance: bool = False) -> torch.Tensor:
        parent_embed = self.get_parent_embed(
            input_ids=x["parent"]["input_ids"],
            attention_mask=x["parent"]["attention_mask"])
        child_embed = self.get_child_embed(
            input_ids=x["child"]["input_ids"],
            attention_mask=x["child"]["attention_mask"])
        res = batch_dot(parent_embed, child_embed)
        return dot_as_distance(res) if as_distance else res


EitherModel = Model | BaselineModel


class TrainingHarness(nn.Module):
    def __init__(self, model: EitherModel) -> None:
        super().__init__()
        self._model = model
        self._softmax = nn.Softmax(dim=1)
        self._loss = nn.BCELoss()
        self._score_loss = nn.MSELoss()

    def get_version(self) -> int:
        return self._model.get_version()

    def score_loss(
            self,
            inputs: TokenizedInput,
            scores: torch.Tensor,
            *,
            as_distance: bool) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._model(inputs, as_distance=as_distance)
        dists = dot_as_distance(scores) if as_distance else scores
        preds = torch.cdist(out, dists)
        return preds, self._score_loss(out, dists)

    def forward(
            self,
            left: TokenizedInput,
            right: TokenizedInput,
            labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_left = self._model(left)
        out_right = self._model(right)
        preds = self._softmax(torch.hstack((out_left, out_right)))
        return preds, self._loss(preds, labels)


def load_model(
        fin: IO[bytes],
        version: int,
        is_harness: bool) -> EitherModel:
    device = get_device()
    model: EitherModel
    if version < 0:
        model = BaselineModel(version=version, ignore_pretrained_warning=True)
    else:
        model = Model(version=version, ignore_pretrained_warning=True)
    if is_harness:
        harness = TrainingHarness(model)
        harness.load_state_dict(torch.load(fin, map_location=device))
    else:
        model.load_state_dict(torch.load(fin, map_location=device))
    return model


class TransformerEmbedding(EmbeddingProvider):
    def __init__(
            self,
            model: EitherModel,
            method: str,
            role: ProviderRole,
            embedding_name: str,
            embedding_hash: str,
            storage_method: StorageMethod) -> None:
        super().__init__(
            method,
            role,
            embedding_name,
            embedding_hash,
            model.get_version(),
            storage_method)
        model.to(get_device())
        self._model = model
        self._tokenizer = get_tokenizer()
        self._is_parent = role == PROVIDER_PARENT

    def get_embedding(self, msg: Message) -> torch.Tensor:
        text = msg.get_text()
        input_obj = self._tokenizer([text])
        self._model.eval()
        with torch.no_grad():
            if self._is_parent:
                return safe_ravel(self._model.get_parent_embed(
                    input_ids=input_obj["input_ids"],
                    attention_mask=input_obj["attention_mask"])).cpu()
            return safe_ravel(self._model.get_child_embed(
                input_ids=input_obj["input_ids"],
                attention_mask=input_obj["attention_mask"])).cpu()

    @staticmethod
    def num_dimensions() -> int:
        return EMBED_SIZE


def load_providers(
        root: str,
        fname: str,
        version: int,
        is_harness: bool) -> EmbeddingProviderMap:
    model_file = os.path.join(root, fname)
    model_hash = get_file_hash(model_file)
    with open_read(model_file, text=False) as fin:
        model = load_model(fin, version, is_harness)
    model_name = fname
    rix = model_name.rfind(".")
    if rix >= 0:
        model_name = model_name[:rix]
    return {
        "parent": TransformerEmbedding(
            model,
            "transformer",
            PROVIDER_PARENT,
            model_name,
            model_hash,
            STORAGE_ARRAY),
        "child": TransformerEmbedding(
            model,
            "transformer",
            PROVIDER_CHILD,
            model_name,
            model_hash,
            STORAGE_ARRAY),
    }


def load_db_providers(
        db: DBConnector,
        model_hash: str,
        storage_method: StorageMethod) -> EmbeddingProviderMap:
    with read_db_model(db, model_hash) as ctx:
        fin, model_name, version, is_harness = ctx
        model = load_model(fin, version, is_harness)
        res: EmbeddingProviderMap = {
            "parent": TransformerEmbedding(
                model,
                "transformer",
                PROVIDER_PARENT,
                model_name,
                model_hash,
                storage_method),
            "child": TransformerEmbedding(
                model,
                "transformer",
                PROVIDER_CHILD,
                model_name,
                model_hash,
                storage_method),
        }
    return res


def get_model_filename_tuple(
        harness: TrainingHarness,
        folder: str,
        *,
        is_cuda: bool,
        ftype: str,
        epoch: int | None,
        ext: str = ".pkl") -> tuple[str, str, str]:
    postfix = "_lg" if is_cuda else ""
    version_tag = f"_v{harness.get_version()}"
    out_pre = f"{ftype}{version_tag}{postfix}_"
    out_post = ext
    return (
        os.path.join(
            folder,
            f"{out_pre}{'' if epoch is None else epoch}{out_post}"),
        out_pre,
        out_post,
    )


def get_model_filename(
        harness: TrainingHarness,
        folder: str,
        *,
        is_cuda: bool,
        ftype: str,
        epoch: int | None,
        ext: str = ".pkl") -> str:
    return get_model_filename_tuple(
        harness, folder, is_cuda=is_cuda, ftype=ftype, epoch=epoch, ext=ext)[0]


def get_epoch_and_load(
        harness: TrainingHarness,
        folder: str,
        *,
        ftype: str,
        is_cuda: bool,
        device: torch.device,
        force_restart: bool) -> tuple[tuple[str, int] | None, int]:
    _, spre, spost = get_model_filename_tuple(
        harness, folder, ftype=ftype, is_cuda=is_cuda, epoch=None)
    mprev = highest_number(listdir(folder), prefix=spre, postfix=spost)
    if not force_restart and mprev is not None:
        prev_fname, prev_epoch = mprev
        harness.load_state_dict(torch.load(
            os.path.join(folder, prev_fname), map_location=device))
        epoch_offset = prev_epoch + 1
    else:
        epoch_offset = 0
    return mprev, epoch_offset


def limit_epoch_data(
        harness: TrainingHarness,
        folder: str,
        *,
        ftype: str,
        is_cuda: bool,
        ext: str,
        count: int) -> None:
    _, spre, spost = get_model_filename_tuple(
        harness, folder, ftype=ftype, is_cuda=is_cuda, epoch=None, ext=ext)

    def load_stats(fname: str) -> EpochStats:
        with open_read(fname, text=True) as fin:
            return cast(EpochStats, json_load(fin))

    def stats_key(
            stats_tup: tuple[str, EpochStats]) -> tuple[float, float, float]:
        _, stats = stats_tup
        return (
            stats["train_val_acc"],
            stats["test_acc"],
            stats["train_acc"],
        )

    def get_stats() -> Iterable[tuple[str, EpochStats]]:
        for fname, _ in sorted(extract_number(
                listdir(folder), spre, spost), key=lambda elem: elem[1]):
            filename = os.path.join(folder, fname)
            stats = load_stats(filename)
            yield filename, stats

    best, to_delete = retain_some(get_stats(), count, key=stats_key)
    for (stats_file, stats) in to_delete:
        print(f"removing {stats_file}")
        remove_file(stats_file)
        print(f"removing {stats['fname']}")
        remove_file(stats["fname"])
    if best:
        best_stats = best[-1][1]
        print(f"best model: {best_stats['fname']}")
        print(f"best train: {best_stats['train_acc']}")
        print(f"best train val: {best_stats['train_val_acc']}")
        print(f"best test: {best_stats['test_acc']}")
