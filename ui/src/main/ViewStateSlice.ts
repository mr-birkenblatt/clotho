import { createSlice } from '@reduxjs/toolkit';
import { GraphView, initView } from '../misc/GraphView';

type ViewState = {
  currentView: Readonly<GraphView>;
  currentChanges: Readonly<number>;
};

type SetAction = {
  payload: {
    view: Readonly<GraphView>;
    changes: number;
    progress: boolean;
  };
};

type ViewReducers = {
  setView: (state: ViewState, action: SetAction) => void;
};

const viewStateSlice = createSlice<ViewState, ViewReducers, string>({
  name: 'viewState',
  initialState: {
    currentView: initView(undefined, undefined),
    currentChanges: 0,
  },
  reducers: {
    setView: (state, action) => {
      const { view, changes, progress } = action.payload;
      if (progress) {
        if (changes !== state.currentChanges) {
          return;
        }
      } else {
        state.currentChanges =
          (Math.max(state.currentChanges, changes) + 1) % 100;
      }
      state.currentView = view;
    },
  },
});

export const { setView } = viewStateSlice.actions;

export default viewStateSlice.reducer;
