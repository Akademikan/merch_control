"""Модель данных (SQLAlchemy 2.0). По умолчанию SQLite, для работы — PostgreSQL."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


def aware(dt: datetime) -> datetime:
    """SQLite не хранит часовой пояс: считаем, что в БД лежит UTC."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Store(Base):
    __tablename__ = "stores"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    radius_m: Mapped[float] = mapped_column(Float, default=60.0)
    tz: Mapped[str] = mapped_column(String(64), default="Europe/Moscow")


class Visit(Base):
    __tablename__ = "visits"
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey("stores.id"))
    merchandiser: Mapped[str] = mapped_column(String(100))  # идентификатор, не ФИО
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    required_skus: Mapped[dict] = mapped_column(JSON, default=dict)  # {"sku": мин. фейсингов}

    store: Mapped[Store] = relationship()
    photos: Mapped[list["Photo"]] = relationship(back_populates="visit", order_by="Photo.id")


class Photo(Base):
    __tablename__ = "photos"
    id: Mapped[int] = mapped_column(primary_key=True)
    visit_id: Mapped[int] = mapped_column(ForeignKey("visits.id"))
    path: Mapped[str] = mapped_column(String(500))
    phash: Mapped[str] = mapped_column(String(32), index=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    meta: Mapped[dict] = mapped_column(JSON)
    checks: Mapped[dict] = mapped_column(JSON)
    detections: Mapped[list] = mapped_column(JSON, default=list)
    completeness: Mapped[float | None] = mapped_column(Float, nullable=True)
    probability: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(16))  # accept | reject | manual
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    supervisor_decision: Mapped[str | None] = mapped_column(String(16), nullable=True)  # разметка для обучения

    visit: Mapped[Visit] = relationship(back_populates="photos")


def make_session_factory(database_url: str):
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
