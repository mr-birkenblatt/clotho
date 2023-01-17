import sqlalchemy as sa
from sqlalchemy.orm import registry
from sqlalchemy.orm.decl_api import DeclarativeMeta

from system.namespace.load import NS_NAME_MAX_LEN
from system.namespace.module import MODULE_MAX_LEN


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
