--[[
  Utils preprended to other lua scripts (redis does not allow require)
--]]
local Utils = {}

-- Get full key for named object
function Utils.get_obj_name(prefix, name, key)
  return prefix .. ":objects:" .. name .. ":" .. key
end

-- Get normal random value -- make sure to call math.randomseed(seed)
-- seed should be computed via time.monotonic() * 1e7
-- function Utils.gaussian(mean, variance)
--   -- https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
--   local u = 0
--   while u < 1e-12 do
--     u = math.random()
--   end
--   local r = math.sqrt(-2 * math.log(u))
--   local th = math.cos(math.rad(math.random()))
--   return variance * r * th + mean
-- end

-- function Utils.get_pexpire(expire_mean, expire_var)
--   local rand = Utils.gaussian(expire_mean, expire_var)
--   local res = math.min(math.max(rand, expire_mean * 0.75), expire_mean * 1.5)
--   return math.floor(res * 1000)
-- end

--[[
  end of utils
--]]
