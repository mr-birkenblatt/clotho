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
    vOrder: [
      "9709aa3742acc01b0247eac2968ae4e4605ef0814c541c1df418309b76fce89d",
      "7264c7cd248f8c907e63d2bc1d0dd25fd1a2e7c5155058699289cea8ca18674d",
    ],
    vCurrentIx: 1,
    vCorrection: 0,
    vOffset: 0,
    vFocus: 0,
    vFocusSmooth: false,
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
      if (vIndex === state.vCurrentIx) {
        return;
      }
      lockLine(state, isParent, lineName, hIndex, false);
      state.vCurrentIx = vIndex;
      state.vOffset = vIndex - 1;
      state.vFocus = vIndex;
      state.vFocusSmooth = false;
    },
    addLine: (state, action) => {
      const { lineName, isBack } = action.payload;
      if (isBack) {
        state.vOrder.push(lineName);
      } else {
        state.vOrder = [lineName, ...state.vOrder];
        state.vCorrection += 1;
      }
    },
    focusV: (state, action) => {
      const { focus } = action.payload;
      state.vFocus = focus;
      state.vFocusSmooth = true;
    },
  },
});

export const {
  addLine,
  focusAt,
  focusV,
  lockCurrent,
  setHCurrentIx,
  setVCurrentIx,
} = lineStateSlice.actions;

export default lineStateSlice.reducer;
