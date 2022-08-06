import { createSlice } from '@reduxjs/toolkit';

export function constructKey(lineName) {
  return `${lineName}`;
}

function lockLine(state, isParent, lineName, adjustedIndex, skipItem) {
  const key = constructKey(lineName);
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
}

export const lineStateSlice = createSlice({
  name: 'lineState',
  initialState: {
    currentLineIxs: {},
    currentLineFocus: {},
    locks: {},
    vOrder: ["L0"],
    vCurrentIx: 0,
    vOffset: 0,
    vPadSize: 0,
    vSize: 450,
  },
  reducers: {
    setHCurrentIx: (state, action) => {
      const { lineName, index } = action.payload;
      state.currentLineIxs[constructKey(lineName)] = index;
    },
    focusAt: (state, action) => {
      const { lineName, index } = action.payload;
      state.currentLineFocus[constructKey(lineName)] = index;
    },
    lockCurrent: (state, action) => {
      const { isParent, lineName, adjustedIndex, skipItem } = action.payload;
      return lockLine(state, isParent, lineName, adjustedIndex, skipItem);
    },
    setVCurrentIx: (state, action) => {
      const { vIndex, hIndex, isParent, lineName } = action.payload;
      lockLine(state, isParent, lineName, hIndex, false);
      state.vCurrentIx = vIndex;
      console.log(`vIndex ${vIndex}`);
    },
    addLine: (state, action) => {
      const { lineName, isBack } = action.payload;
      console.log(`addLine ${lineName} ${isBack}`);
      if (isBack) {
        console.log(`added line ${lineName}`);
        state.vOrder.push(lineName);
      } else {
        state.vOrder = [lineName, ...state.vOrder];
        state.vOffset += 1;
      }
    },
  },
});

export const {
  addLine,
  focusAt,
  lockCurrent,
  setHCurrentIx,
  setVCurrentIx,
} = lineStateSlice.actions;

export default lineStateSlice.reducer;
