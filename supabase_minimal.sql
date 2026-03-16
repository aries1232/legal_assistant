-- Supabase schema for legal_assistant
-- UUID-first schema. Uses dedicated chat table `legal_chat_messages`
-- so it does not conflict with any pre-existing `messages` table.

create extension if not exists pgcrypto;

-- 1) Chat sessions
create table if not exists public.sessions (
  id uuid primary key default gen_random_uuid(),
  created_at timestamptz not null default now()
);

-- 2) Uploaded documents (Document Library)
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  session_id uuid null,
  filename text not null,
  file_path text not null,
  file_size_bytes bigint not null default 0,
  ingest_status text not null default 'processing'
    check (ingest_status in ('processing', 'ready', 'failed')),
  chunk_count int not null default 0,
  ingest_error text null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 3) Chat messages (app-specific table)
create table if not exists public.legal_chat_messages (
  id bigserial primary key,
  session_id uuid not null,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  sources jsonb not null default '[]'::jsonb,
  selected_doc_ids jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

-- If tables already existed with text IDs, migrate to UUID types.
alter table public.sessions
  alter column id type uuid using id::uuid;

alter table public.documents
  alter column id type uuid using id::uuid,
  alter column session_id type uuid using nullif(session_id::text, '')::uuid;

alter table public.legal_chat_messages
  alter column session_id type uuid using nullif(session_id::text, '')::uuid;

-- Recreate FKs with matching UUID types.
alter table public.documents
  drop constraint if exists documents_session_id_fkey;

alter table public.documents
  add constraint documents_session_id_fkey
  foreign key (session_id) references public.sessions(id) on delete set null;

alter table public.legal_chat_messages
  drop constraint if exists legal_chat_messages_session_id_fkey;

alter table public.legal_chat_messages
  add constraint legal_chat_messages_session_id_fkey
  foreign key (session_id) references public.sessions(id) on delete cascade;

-- Keep updated_at current on document updates.
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_documents_set_updated_at on public.documents;
create trigger trg_documents_set_updated_at
before update on public.documents
for each row execute function public.set_updated_at();

-- Helpful indexes.
create index if not exists idx_messages_session_created
  on public.legal_chat_messages(session_id, created_at);

create index if not exists idx_documents_created
  on public.documents(created_at desc);

create index if not exists idx_documents_session_created
  on public.documents(session_id, created_at desc);

-- Private storage bucket for uploaded PDFs.
insert into storage.buckets (id, name, public)
values ('legal-docs', 'legal-docs', false)
on conflict (id) do nothing;
