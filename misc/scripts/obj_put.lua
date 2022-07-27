--[[
  1. Sets key with val
  2. Sets expiration for key if ttl exists
--]]
local preserve_expire = ARGV[1]
local val = ARGV[2]
local key = KEYS[1]

-- Note: preserve_expire expected to be "1" or "0" (=~ T/F)
if preserve_expire == "1" then
  local pexpire = tonumber(redis.call("PTTL", key))
  if pexpire > 0 then
    redis.call("SET", key, val, "PX", pexpire)
    -- pexpire == -2 -> key does not exist
    -- pexpire == -1 -> key has no expire
    -- pexpire == 0 -> treated as expired
    -- pexpire > 0 -> key has expiration set
  end
else
  redis.call("SET", key, val)
end
