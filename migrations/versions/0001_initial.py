"""Esquema inicial do agente Gomme Fenice (consolidado a partir do agente anterior, sem as colunas
especificas dela: lojas, horarios por dia e janela de contexto anterior).

Revision ID: 0001
Revises:
Create Date: 2026-09-30

"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("session_id", sa.String(), primary_key=True),
        sa.Column("environment", sa.String(), nullable=False, server_default="producao"),
        sa.Column("contact_id", sa.String(), nullable=False),
        sa.Column("phone", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("stage", sa.String(), nullable=False, server_default="ativo"),
        sa.Column("slots", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("transferred_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("paused_by_human", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(), sa.ForeignKey("conversations.session_id"), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cache_read_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cache_write_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("audio_cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("audio_provider", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"])
    op.create_index("ix_messages_created_at", "messages", ["created_at"])

    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(), sa.ForeignKey("conversations.session_id"), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_events_session_id", "events", ["session_id"])
    op.create_index("ix_events_created_at", "events", ["created_at"])

    op.create_table(
        "app_settings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("active_provider", sa.String(), nullable=False, server_default="gemini"),
        sa.Column("active_model", sa.String(), nullable=False, server_default="gemini-2.5-flash"),
        sa.Column("provider_api_keys_encrypted", sa.Text(), nullable=True),
        sa.Column("system_prompt_static", sa.Text(), nullable=True),
        sa.Column("tts_model_id", sa.String(), nullable=True),
        sa.Column("audio_replies_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("helena_credentials_encrypted", sa.Text(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "knowledge_entries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("label", sa.String(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_knowledge_entries_created_at", "knowledge_entries", ["created_at"])

    op.create_table(
        "test_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("sender_phone", sa.String(), nullable=False),
        sa.Column("receiver_phone", sa.String(), nullable=False),
        sa.Column("activation_code", sa.String(), nullable=False),
        sa.Column("persona_prompt", sa.Text(), nullable=False),
        sa.Column("target_session_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="running"),
        sa.Column("turns_done", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_turns", sa.Integer(), nullable=False, server_default="8"),
        sa.Column("llm_provider", sa.String(), nullable=False, server_default=""),
        sa.Column("llm_model", sa.String(), nullable=False, server_default=""),
        sa.Column("total_cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("test_runs")
    op.drop_index("ix_knowledge_entries_created_at", table_name="knowledge_entries")
    op.drop_table("knowledge_entries")
    op.drop_table("app_settings")
    op.drop_index("ix_events_created_at", table_name="events")
    op.drop_index("ix_events_session_id", table_name="events")
    op.drop_table("events")
    op.drop_index("ix_messages_created_at", table_name="messages")
    op.drop_index("ix_messages_session_id", table_name="messages")
    op.drop_table("messages")
    op.drop_table("conversations")
