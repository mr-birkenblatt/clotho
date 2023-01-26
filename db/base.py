import numpy as np
import sqlalchemy as sa
from psycopg2.extensions import AsIs, register_adapter
from sqlalchemy.orm import registry
from sqlalchemy.orm.decl_api import DeclarativeMeta

from system.msgs.message import MHash
from system.namespace.load import NS_NAME_MAX_LEN
from system.namespace.module import MODULE_MAX_LEN


def adapt_numpy_float64(numpy_float64: np.float64) -> AsIs:
    return AsIs(numpy_float64)


def adapt_numpy_int64(numpy_int64: np.int64) -> AsIs:
    return AsIs(numpy_int64)


register_adapter(np.float64, adapt_numpy_float64)
register_adapter(np.int64, adapt_numpy_int64)


mapper_registry = registry()


class Base(
        metaclass=DeclarativeMeta):  # pylint: disable=too-few-public-methods
    __abstract__ = True
    __table__: sa.Table

    registry = mapper_registry
    metadata = mapper_registry.metadata

    __init__ = mapper_registry.constructor


class NamespaceTable(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "namespace"

    id = sa.Column(
        sa.Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False,
        unique=True)
    name = sa.Column(
        sa.String(NS_NAME_MAX_LEN),
        primary_key=True,
        nullable=False,
        unique=True)


class ModulesTable(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "modules"

    module = sa.Column(
        sa.String(MODULE_MAX_LEN),
        primary_key=True,
        nullable=False,
        unique=True)
    version = sa.Column(
        sa.Integer,
        nullable=False)


class MHashTable(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "mhashes"

    id = sa.Column(
        sa.Integer,
        primary_key=True,
        autoincrement=True,
        nullable=False,
        unique=True)
    mhash = sa.Column(
        sa.String(MHash.parse_size()),
        nullable=False,
        unique=True)

    idx_mhash = sa.Index("mhash")
