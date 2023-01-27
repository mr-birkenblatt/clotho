-- This script assumes module version 1 for the modules msgs, embed:dbcache
-- Create mhashes table before running queries here

-- copy all hashes into the new table
INSERT INTO mhashes (mhash)
    (SELECT DISTINCT mhash FROM msgs)
    ON CONFLICT (mhash) DO NOTHING;

INSERT INTO mhashes (mhash)
    (SELECT DISTINCT mhash FROM topics)
    ON CONFLICT (mhash) DO NOTHING;

INSERT INTO mhashes (mhash)
    (SELECT DISTINCT mhash FROM embed)
    ON CONFLICT (mhash) DO NOTHING;

-- verify that mhashes is not empty
SELECT COUNT(*) FROM mhashes;

-- verify that there are no duplicates
SELECT t.cnt as count, t.mhash FROM (
        SELECT COUNT(*) as cnt, mhash
        FROM mhashes GROUP BY mhash) as t
    WHERE t.cnt > 2 ORDER BY t.cnt DESC;

-- *** msgs ***
-- look at msgs
SELECT * FROM msgs LIMIT 20;

-- add foreign column to msgs table
ALTER TABLE msgs ADD COLUMN mhash_id integer NOT NULL DEFAULT 0;

-- properly set foreign keys
UPDATE msgs
    SET mhash_id = mhashes.id
    FROM mhashes
    WHERE msgs.mhash = mhashes.mhash;

-- add foreign constraint
ALTER TABLE msgs
    ADD FOREIGN KEY (mhash_id)
    REFERENCES mhashes (id)
        MATCH SIMPLE
        ON UPDATE CASCADE
        ON DELETE CASCADE;

-- remove default
ALTER TABLE msgs ALTER COLUMN mhash_id DROP DEFAULT;

-- remove primary key
ALTER TABLE msgs DROP CONSTRAINT msgs_pkey;

-- add new primary key
ALTER TABLE msgs ADD PRIMARY KEY (namespace_id, mhash_id);

-- check keys
SELECT s.namespace_id, s.mhash_id, s.mhash as s_mhash, m.mhash as m_mhash
    FROM msgs as s, mhashes as m
    WHERE s.mhash_id = m.id
    LIMIT 10;

-- remove mhash column
ALTER TABLE msgs DROP COLUMN mhash;

-- *** end of msgs ***

-- *** topics ***
-- look at topics
SELECT * FROM topics LIMIT 20;

-- add foreign column to topics table
ALTER TABLE topics ADD COLUMN mhash_id integer NOT NULL DEFAULT 0;

-- properly set foreign keys
UPDATE topics
    SET mhash_id = mhashes.id
    FROM mhashes
    WHERE topics.mhash = mhashes.mhash;

-- add foreign constraint
ALTER TABLE topics
    ADD FOREIGN KEY (mhash_id)
    REFERENCES mhashes (id)
        MATCH SIMPLE
        ON UPDATE CASCADE
        ON DELETE CASCADE;

-- remove default
ALTER TABLE topics ALTER COLUMN mhash_id DROP DEFAULT;

-- remove primary key
ALTER TABLE topics DROP CONSTRAINT topics_pkey;

-- add new primary key
ALTER TABLE topics ADD PRIMARY KEY (namespace_id, id, mhash_id);

-- check keys
SELECT t.namespace_id, t.id, t.mhash_id, t.mhash as t_mhash, m.mhash as m_mhash
    FROM topics as t, mhashes as m
    WHERE t.mhash_id = m.id
    LIMIT 10;

-- remove mhash column
ALTER TABLE topics DROP COLUMN mhash;

-- *** end of topics ***

-- *** embed ***
-- look at embed
SELECT * FROM embed LIMIT 20;

-- add foreign column to embed table
ALTER TABLE embed ADD COLUMN mhash_id integer NOT NULL DEFAULT 0;

-- properly set foreign keys
UPDATE embed
    SET mhash_id = mhashes.id
    FROM mhashes
    WHERE embed.mhash = mhashes.mhash;

-- add foreign constraint
ALTER TABLE embed
    ADD FOREIGN KEY (mhash_id)
    REFERENCES mhashes (id)
        MATCH SIMPLE
        ON UPDATE CASCADE
        ON DELETE CASCADE;

-- remove default
ALTER TABLE embed ALTER COLUMN mhash_id DROP DEFAULT;

-- remove primary key
ALTER TABLE embed DROP CONSTRAINT embed_pkey;

-- add new primary key
ALTER TABLE embed ADD PRIMARY KEY (config_id, mhash_id);

-- check keys
SELECT e.config_id, e.mhash_id, e.mhash as e_mhash, m.mhash as m_mhash
    FROM embed as e, mhashes as m
    WHERE e.mhash_id = m.id
    LIMIT 10;

-- remove mhash column
ALTER TABLE embed DROP COLUMN mhash;

SELECT * FROM embed WHERE config_id = 1 ORDER BY main_order ASC LIMIT 10;

-- *** end of embed ***

-- *** embedconfig and models ***
-- look at models
SELECT * FROM models LIMIT 20;

-- add sequence
CREATE SEQUENCE IF NOT EXISTS models_id_seq
    INCREMENT 1
    START 1
    MINVALUE 1
    MAXVALUE 2147483647
    CACHE 1;

-- add id column
ALTER TABLE models ADD COLUMN
    id integer NOT NULL
    DEFAULT nextval('models_id_seq'::regclass);

-- connect sequence to table
ALTER SEQUENCE models_id_seq OWNED BY models.id;

-- make id unqiue
ALTER TABLE models ADD CONSTRAINT models_id_key UNIQUE (id);

-- look at embedconfig
SELECT * FROM embedconfig LIMIT 20;

-- add foreign column to embedconfig table
ALTER TABLE embedconfig ADD COLUMN model_id integer NOT NULL DEFAULT 0;

-- properly set foreign keys
UPDATE embedconfig
    SET model_id = models.id
    FROM models
    WHERE embedconfig.model_hash = models.model_hash;

-- add foreign constraint
ALTER TABLE embedconfig
    ADD FOREIGN KEY (model_id)
    REFERENCES models (id)
        MATCH SIMPLE
        ON UPDATE CASCADE
        ON DELETE CASCADE;

-- remove default
ALTER TABLE embedconfig ALTER COLUMN model_id DROP DEFAULT;

-- remove primary key
ALTER TABLE embedconfig DROP CONSTRAINT embedconfig_pkey;

-- add new primary key
ALTER TABLE embedconfig ADD PRIMARY KEY (namespace_id, role, model_id);

-- check keys
SELECT
        e.namespace_id,
        e.role,
        e.config_id,
        e.model_id,
        e.model_hash as e_model_hash,
        m.model_hash as m_model_hash
    FROM embedconfig as e, models as m
    WHERE e.model_id = m.id
    LIMIT 10;

-- remove model_hash column
ALTER TABLE embedconfig DROP COLUMN model_hash;

-- remove models primary key
ALTER TABLE models DROP CONSTRAINT models_pkey;

-- add new models primary key
ALTER TABLE models ADD PRIMARY KEY (id);

SELECT e.*, m.model_hash
    FROM embedconfig as e, models as m
    WHERE e.model_id = m.id
    LIMIT 10;

SELECT * FROM models;

-- *** end of embedconfig ***
