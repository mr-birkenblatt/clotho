import { createSlice } from '@reduxjs/toolkit';
import {
  AdjustedLineIndex,
  LineKey,
  MHash,
  TOPIC_KEY,
} from '../misc/CommentGraph';
import { num, safeStringify } from '../misc/util';

export type LineLock = {
  lineKey: Readonly<LineKey>;
  mhash: Readonly<MHash> | undefined;
};

export type LineIndex = number & { _lineIndex: void };

export const LOCK_INDEX = -1 as LineIndex;

export type VIndex = number & { _vIndex: void };
export type VCorrection = number & { _vCorrection: void };
export type VArrIndex = number & { _vArrIndex: void };
export type VOffset = number & { _vOffset: void };

export function constructKey(lineKey: Readonly<LineKey>): string {
  return safeStringify(lineKey);
}

type LineState = {
  currentLineIxs: { [key: string]: Readonly<LineIndex> };
  currentLineFocus: { [key: string]: Readonly<LineIndex> };
  locks: { [key: string]: Readonly<LineLock> };
  lockIndex: { [key: string]: Readonly<AdjustedLineIndex> | undefined };
  vOrder: LineKey[];
  vCurrentIx: Readonly<VIndex>;
  vCorrection: Readonly<VCorrection>;
  vOffset: Readonly<VOffset>;
  vFocus: Readonly<VIndex>;
  vFocusSmooth: boolean;
};

type AddAction = {
  payload: {
    lineKey: Readonly<LineKey>;
    isBack: boolean;
  };
};

type IndexAction = {
  payload: {
    lineKey: Readonly<LineKey>;
    index: Readonly<LineIndex>;
  };
};

type LockAction = {
  payload: {
    lineKey: Readonly<LineKey>;
    mhash: Readonly<MHash> | undefined;
  };
};

type LockIndexAction = {
  payload: {
    lineKey: Readonly<LineKey>;
    lockIndex: Readonly<AdjustedLineIndex>;
  };
};

type SetVAction = {
  payload: {
    vIndex: Readonly<VIndex>;
    mhash: Readonly<MHash> | undefined;
    lineKey: Readonly<LineKey>;
  };
};

type FocusVAction = {
  payload: {
    focus: Readonly<VIndex>;
  };
};

type LineReducers = {
  addLine: (state: LineState, action: AddAction) => void;
  focusAt: (state: LineState, action: IndexAction) => void;
  focusV: (state: LineState, action: FocusVAction) => void;
  lockCurrent: (state: LineState, action: LockAction) => void;
  setLockIndex: (state: LineState, action: LockIndexAction) => void;
  setHCurrentIx: (state: LineState, action: IndexAction) => void;
  setVCurrentIx: (state: LineState, action: SetVAction) => void;
};

function lockLine(
  state: LineState,
  lineKey: Readonly<LineKey>,
  mhash: Readonly<MHash> | undefined,
) {
  const key = constructKey(lineKey);
  const { currentLineIxs, currentLineFocus, locks, lockIndex } = state;
  if (currentLineIxs[key] === LOCK_INDEX) {
    return;
  }
  const locked: LineLock = {
    lineKey,
    mhash,
  };
  locks[key] = locked;
  lockIndex[key] = undefined;
  currentLineIxs[key] = LOCK_INDEX;
  currentLineFocus[key] = LOCK_INDEX;
}

export const lineStateSlice = createSlice<LineState, LineReducers, string>({
  name: 'lineState',
  initialState: {
    currentLineIxs: {},
    currentLineFocus: {},
    locks: {},
    lockIndex: {},
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
      const { lineKey, mhash } = action.payload;
      return lockLine(state, lineKey, mhash);
    },
    setLockIndex: (state, action) => {
      const { lineKey, lockIndex } = action.payload;
      const key = constructKey(lineKey);
      state.lockIndex[key] = lockIndex;
    },
    setVCurrentIx: (state, action) => {
      const { vIndex, mhash, lineKey } = action.payload;
      if (vIndex === state.vCurrentIx) {
        return;
      }
      lockLine(state, lineKey, mhash);
      const vOrder = state.vOrder;
      const oldArrIx = (num(state.vCurrentIx) +
        num(state.vCorrection)) as VCorrection;
      const oldRemain = vOrder.length - oldArrIx;
      const arrIx = (num(vIndex) + num(state.vCorrection)) as VCorrection;
      if (arrIx < 0 && arrIx >= vOrder.length) {
        console.warn('new index out of bounds', vIndex, arrIx, vOrder);
        return;
      }
      state.vCurrentIx = vIndex;
      const fromIx = Math.max(0, arrIx - 1);
      const toIx = fromIx + oldRemain;
      state.vOffset = (num(vIndex) - arrIx + fromIx) as VOffset;
      state.vCorrection = -state.vOffset as VCorrection;
      state.vOrder = vOrder.slice(fromIx, toIx);
      state.vFocus = vIndex;
      state.vFocusSmooth = false;
    },
    addLine: (state, action) => {
      const { lineKey, isBack } = action.payload;
      if (isBack) {
        state.vOrder.push(lineKey);
      } else {
        state.vOrder = [lineKey, ...state.vOrder];
        state.vOffset = (num(state.vOffset) - 1) as VOffset;
        state.vCorrection = (num(state.vCorrection) + 1) as VCorrection;
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
  setLockIndex,
  setHCurrentIx,
  setVCurrentIx,
} = lineStateSlice.actions;

export default lineStateSlice.reducer;
