import { createSlice } from '@reduxjs/toolkit';

export type LineLock = {
  isParent: boolean;
  lineName: string;
  index: number;
  skipItem: boolean;
};

type LineState = {
  currentLineIxs: { [key: string]: number };
  currentLineFocus: { [key: string]: number };
  locks: { [key: string]: LineLock };
  vOrder: string[];
  vCurrentIx: number;
  vCorrection: number;
  vOffset: number;
  vFocus: number;
  vFocusSmooth: boolean;
};

type AddAction = {
  payload: {
    lineName: string;
    isBack: boolean;
  };
};

type IndexAction = {
  payload: {
    lineName: string;
    index: number;
  };
};

type LockAction = {
  payload: {
    isParent: boolean;
    lineName: string;
    adjustedIndex: number;
    skipItem: boolean;
  };
};

type SetVAction = {
  payload: {
    vIndex: number;
    hIndex: number;
    isParent: boolean;
    lineName: string;
  };
};

type FocusVAction = {
  payload: {
    focus: number;
  };
};

type LineReducers = {
  addLine: (state: LineState, action: AddAction) => void;
  focusAt: (state: LineState, action: IndexAction) => void;
  focusV: (state: LineState, action: FocusVAction) => void;
  lockCurrent: (state: LineState, action: LockAction) => void;
  setHCurrentIx: (state: LineState, action: IndexAction) => void;
  setVCurrentIx: (state: LineState, action: SetVAction) => void;
};

export function constructKey(lineName: string): string {
  return `${lineName}`;
}

function lockLine(
  state: LineState,
  isParent: boolean,
  lineName: string,
  adjustedIndex: number,
  skipItem: boolean
) {
  const key = constructKey(lineName);
  const { currentLineIxs, currentLineFocus, locks } = state;
  if (currentLineIxs[key] < 0) {
    return;
  }
  const locked: LineLock = {
    isParent,
    lineName,
    index: adjustedIndex,
    skipItem,
  };
  locks[key] = locked;
  currentLineIxs[key] = -1;
  currentLineFocus[key] = -1;
}

export const lineStateSlice = createSlice<LineState, LineReducers, string>({
  name: 'lineState',
  initialState: {
    currentLineIxs: {},
    currentLineFocus: {},
    locks: {},
    vOrder: [
      '!0',
      '9709aa3742acc01b0247eac2968ae4e4605ef0814c541c1df418309b76fce89d',
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
