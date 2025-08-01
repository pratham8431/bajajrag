from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, JSON, DateTime, ForeignKey, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    name         = Column(String, nullable=False)
    url          = Column(Text,  nullable=False)
    ingested_at  = Column(DateTime, default=datetime.utcnow)
    chunks       = relationship("Chunk", back_populates="document")

class Chunk(Base):
    __tablename__ = "chunks"
    id           = Column(String, primary_key=True)  # UUID from our chunker
    document_id  = Column(Integer, ForeignKey("documents.id"), nullable=False)
    page         = Column(Integer)
    section      = Column(String)
    chunk_metadata = Column(JSON)  # Renamed from metadata to avoid SQLAlchemy conflict
    created_at   = Column(DateTime, default=datetime.utcnow)
    document     = relationship("Document", back_populates="chunks")

class QueryLog(Base):
    __tablename__ = "query_logs"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    question     = Column(Text, nullable=False)
    parsed_query = Column(JSON)
    asked_at     = Column(DateTime, default=datetime.utcnow)
    response     = Column(JSON)
