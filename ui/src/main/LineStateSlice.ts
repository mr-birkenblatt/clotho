import { createSlice } from '@reduxjs/toolkit';
import { AdjustedLineIndex, LineKey, TOPIC_KEY } from '../misc/CommentGraph';
import { safeStringify } from '../misc/util';

export type LineLock = {
  lineKey: LineKey;
  index: AdjustedLineIndex;
  skipItem: boolean;
};

export type LineIndex = number & { _lineIndex: void };

export const LOCK_INDEX = -1 as LineIndex;

export type VIndex = number & { _vIndex: void };
export type VCorrection = number & { _vCorrection: void };
export type VOffset = number & { _vOffset: void };

export function constructKey(lineKey: LineKey): string {
  return safeStringify(lineKey);
}

type LineState = {
  currentLineIxs: { [key: string]: LineIndex };
  currentLineFocus: { [key: string]: LineIndex };
  locks: { [key: string]: LineLock };
  vOrder: LineKey[];
  vCurrentIx: VIndex;
  vCorrection: VCorrection;
  vOffset: VOffset;
  vFocus: VIndex;
  vFocusSmooth: boolean;
};

type AddAction = {
  payload: {
    lineKey: LineKey;
    isBack: boolean;
  };
};

type IndexAction = {
  payload: {
    lineKey: LineKey;
    index: LineIndex;
  };
};

type LockAction = {
  payload: {
    lineKey: LineKey;
    adjustedIndex: AdjustedLineIndex;
    skipItem: boolean;
  };
};

type SetVAction = {
  payload: {
    vIndex: VIndex;
    hIndex: AdjustedLineIndex;
    lineKey: LineKey;
  };
};

type FocusVAction = {
  payload: {
    focus: VIndex;
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

function lockLine(
  state: LineState,
  lineKey: LineKey,
  adjustedIndex: AdjustedLineIndex,
  skipItem: boolean,
) {
  const key = constructKey(lineKey);
  const { currentLineIxs, currentLineFocus, locks } = state;
  if (currentLineIxs[key] === LOCK_INDEX) {
    return;
  }
  const locked: LineLock = {
    lineKey,
    index: adjustedIndex,
    skipItem,
  };
  locks[key] = locked;
  currentLineIxs[key] = LOCK_INDEX;
  currentLineFocus[key] = LOCK_INDEX;
}

export const lineStateSlice = createSlice<LineState, LineReducers, string>({
  name: 'lineState',
  initialState: {
    currentLineIxs: {},
    currentLineFocus: {},
    locks: {},
    vOrder: [TOPIC_KEY],
    vCurrentIx: 0 as VIndex,
    vCorrection: 0 as VCorrection,
    vOffset: 0 as VOffset,
    vFocus: 0 as VIndex,
    vFocusSmooth: false,
  },
  reducers: {
    setHCurrentIx: (state, action) => {
      const { lineKey, index } = action.payload;
      state.currentLineIxs[constructKey(lineKey)] = index;
    },
    focusAt: (state, action) => {
      const { lineKey, index } = action.payload;
      state.currentLineFocus[constructKey(lineKey)] = index;
    },
    lockCurrent: (state, action) => {
      const { lineKey, adjustedIndex, skipItem } = action.payload;
      return lockLine(state, lineKey, adjustedIndex, skipItem);
    },
    setVCurrentIx: (state, action) => {
      const { vIndex, hIndex, lineKey } = action.payload;
      if (vIndex === state.vCurrentIx) {
        return;
      }
      lockLine(state, lineKey, hIndex, false);
      state.vCurrentIx = vIndex;
      state.vOffset = vIndex - 1 as VOffset;
      state.vFocus = vIndex;
      state.vFocusSmooth = false;
    },
    addLine: (state, action) => {
      const { lineKey, isBack } = action.payload;
      if (isBack) {
        state.vOrder.push(lineKey);
      } else {
        state.vOrder = [lineKey, ...state.vOrder];
        state.vCorrection = state.vCorrection + 1 as VCorrection;
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
