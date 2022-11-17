import { createSlice } from '@reduxjs/toolkit';
import { GraphView, initView } from '../misc/GraphView';

type ViewState = {
  currentView: Readonly<GraphView>;
};

type SetAction = {
  payload: {
    view: Readonly<GraphView>;
  };
};

type ViewReducers = {
  setView: (state: ViewState, action: SetAction) => void;
};

export const viewStateSlice = createSlice<ViewState, ViewReducers, string>({
  name: 'viewState',
  initialState: {
    currentView: initView(undefined, undefined),
  },
  reducers: {
    setView: (state, action) => {
      state.currentView = action.payload.view;
    },
  },
});

export const { setView } = viewStateSlice.actions;

export default viewStateSlice.reducer;
