import { createSlice } from '@reduxjs/toolkit';

export function constructKey(isParent, lineName) {
  return `${isParent}-${lineName}`;
}

export const lineStateSlice = createSlice({
  name: 'lineState',
  initialState: {
    currentLineIxs: {},
    currentLineFocus: {},
    locks: {},
    order: [],
    currentLine: 0,
  },
  reducers: {
    setCurrentIx: (state, action) => {
      const { isParent, lineName, index } = action.payload;
      state.currentLineIxs[constructKey(isParent, lineName)] = index;
    },
    focusAt: (state, action) => {
      const { isParent, lineName, index } = action.payload;
      state.currentLineFocus[constructKey(isParent, lineName)] = index;
    },
    lockCurrent: (state, action) => {
      const { isParent, lineName, adjustedIndex, skipItem } = action.payload;
      const key = constructKey(isParent, lineName);
      const { currentLineIxs, currentLineFocus, locks } = state;
      if (currentLineIxs[key] < 0) {
        return;
      }
      const locked = {
        isParent,
        lineName,
        index: adjustedIndex,
        skipItem,
      };
      locks[key] = locked;
      currentLineIxs[key] = -1;
      currentLineFocus[key] = -1;
    },
  },
});

export const { lockCurrent, focusAt, setCurrentIx } = lineStateSlice.actions;

export default lineStateSlice.reducer;
