--[[
  Delete all keys matching pattern except for workers in memory
--]]

-- partial match for worker keys
local patterns = cjson.decode(ARGV[1])
for _, pattern in pairs(patterns) do
  local workers = redis.call("KEYS", pattern)
  -- loop over workers
  for _, val in pairs(workers) do
    local found = false
    for ix = 2, #ARGV do
      if val == ARGV[ix] then
        -- skip cached workers
        found = true
        break
      end
    end
    if not found then
      -- del workers that are not cached
      redis.call("DEL", val)
    end
  end
end
