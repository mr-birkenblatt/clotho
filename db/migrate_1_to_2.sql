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
    SET msgs.mhash_id = mhashes.id
    FROM mhashes
    WHERE msgs.mhash = mhashes.mhash;
