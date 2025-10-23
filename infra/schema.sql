-- Postgres schema for Prompt Atlas Engine
create table if not exists users (
  id uuid primary key,
  email text unique,
  plan text not null default 'free',
  created_at timestamptz not null default now()
);
-- (other tables omitted for brevity in this quick zip)
