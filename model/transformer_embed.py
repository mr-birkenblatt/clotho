import os
from typing import Callable, cast, IO, Iterable, Literal, TypedDict

import torch
from torch import nn

# FIXME: add transformer stubs
from transformers import (  # type: ignore
    DistilBertModel,
    DistilBertTokenizer,
    logging,
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
)
from system.embedding.dbcache import read_db_model
from system.msgs.message import Message


EMBED_SIZE = 768


def get_device() -> torch.device:
    is_cuda = torch.cuda.is_available()
    return torch.device("cuda") if is_cuda else torch.device("cpu")


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


class Model(nn.Module):
    def __init__(self, version: int) -> None:
        super().__init__()
        assert version >= 0
        self._bert_parent = DistilBertModel.from_pretrained(
            "distilbert-base-uncased")
        self._bert_child = DistilBertModel.from_pretrained(
            "distilbert-base-uncased")
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

    def forward(self, x: dict[ProviderRole, TokenizedInput]) -> torch.Tensor:
        parent_cls = self.get_parent_embed(
            input_ids=x["parent"]["input_ids"],
            attention_mask=x["parent"]["attention_mask"])
        child_cls = self.get_child_embed(
            input_ids=x["child"]["input_ids"],
            attention_mask=x["child"]["attention_mask"])
        if self._cos is not None:
            return self._cos(parent_cls, child_cls).reshape([-1, 1])
        batch_size = parent_cls.shape[0]
        return torch.bmm(
            parent_cls.reshape([batch_size, 1, -1]),
            child_cls.reshape([batch_size, -1, 1])).reshape([-1, 1])


class BaselineModel(nn.Module):
    def __init__(self, version: int) -> None:
        super().__init__()
        assert version < 0
        self._bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased")
        if version == -2:
            self._agg = AGG_CLS
        else:
            self._agg = AGG_MEAN
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
            attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self._bert(
            input_ids=input_ids, attention_mask=attention_mask)
        return self.get_agg(outputs.last_hidden_state)

    def get_parent_embed(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor) -> torch.Tensor:
        return self._embed(input_ids, attention_mask)

    def get_child_embed(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor) -> torch.Tensor:
        return self._embed(input_ids, attention_mask)

    def forward(self, x: dict[ProviderRole, TokenizedInput]) -> torch.Tensor:
        parent_cls = self.get_parent_embed(
            input_ids=x["parent"]["input_ids"],
            attention_mask=x["parent"]["attention_mask"])
        child_cls = self.get_child_embed(
            input_ids=x["child"]["input_ids"],
            attention_mask=x["child"]["attention_mask"])
        batch_size = parent_cls.shape[0]
        return torch.bmm(
            parent_cls.reshape([batch_size, 1, -1]),
            child_cls.reshape([batch_size, -1, 1])).reshape([-1, 1])


EitherModel = Model | BaselineModel


class TrainingHarness(nn.Module):
    def __init__(self, model: EitherModel) -> None:
        super().__init__()
        self._model = model
        self._softmax = nn.Softmax(dim=1)
        self._loss = nn.BCELoss()

    def get_version(self) -> int:
        return self._model.get_version()

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
    verbosity = logging.get_verbosity()
    try:
        logging.set_verbosity_error()

        device = get_device()
        if version < 0:
            model: EitherModel = BaselineModel(version=version)
        else:
            model = Model(version=version)
        if is_harness:
            harness = TrainingHarness(model)
            harness.load_state_dict(torch.load(fin, map_location=device))
        else:
            model.load_state_dict(torch.load(fin, map_location=device))
        return model
    finally:
        logging.set_verbosity(verbosity)


class TransformerEmbedding(EmbeddingProvider):
    def __init__(
            self,
            model: EitherModel,
            method: str,
            role: ProviderRole,
            embedding_name: str,
            embedding_hash: str) -> None:
        super().__init__(
            method, role, embedding_name, embedding_hash, model.get_version())
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
            model, "transformer", PROVIDER_PARENT, model_name, model_hash),
        "child": TransformerEmbedding(
            model, "transformer", PROVIDER_CHILD, model_name, model_hash),
    }


def load_db_providers(
        db: DBConnector, model_hash: str) -> EmbeddingProviderMap:
    with read_db_model(db, model_hash) as ctx:
        fin, model_name, version, is_harness = ctx
        model = load_model(fin, version, is_harness)
        res: EmbeddingProviderMap = {
            "parent": TransformerEmbedding(
                model, "transformer", PROVIDER_PARENT, model_name, model_hash),
            "child": TransformerEmbedding(
                model, "transformer", PROVIDER_CHILD, model_name, model_hash),
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
