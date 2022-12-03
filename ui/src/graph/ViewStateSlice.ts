import { createSlice } from '@reduxjs/toolkit';
import { UserId } from '../api/types';
import {
  GraphView,
  initUserView,
  initView,
  removeAllLinks,
} from './GraphView';

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

type InitTopicAction = {
  payload: { changes?: number };
};

type InitUserAction = {
  payload: {
    userId: Readonly<UserId>;
    changes?: number;
  };
};

type RefreshLinksAction = {
  payload: { changes?: number };
};

type ViewReducers = {
  setView: (state: ViewState, action: SetAction) => void;
  initTopic: (state: ViewState, action: InitTopicAction) => void;
  initUser: (state: ViewState, action: InitUserAction) => void;
  refreshLinks: (state: ViewState, action: RefreshLinksAction) => void;
};

function incChanges(
  state: Readonly<ViewState>,
  changes: number | undefined,
): number {
  const prev =
    changes !== undefined
      ? Math.max(state.currentChanges, changes)
      : state.currentChanges;
  return (prev + 1) % 100;
}

const viewStateSlice = createSlice<ViewState, ViewReducers, string>({
  name: 'viewState',
  initialState: {
    currentView: initView(undefined, undefined),
    currentChanges: 0,
  },
  reducers: {
    setView: (state, action) => {
      const { view, changes, progress } = action.payload;
      if (progress && changes !== state.currentChanges) {
        return;
      }
      state.currentChanges = incChanges(state, changes);
      state.currentView = view;
    },
    initTopic: (state, action) => {
      const { changes } = action.payload;
      state.currentChanges = incChanges(state, changes);
      state.currentView = initView(undefined, undefined);
    },
    initUser: (state, action) => {
      const { userId, changes } = action.payload;
      state.currentChanges = incChanges(state, changes);
      state.currentView = initUserView(userId);
    },
    refreshLinks: (state, action) => {
      const { changes } = action.payload;
      state.currentChanges = incChanges(state, changes);
      state.currentView = removeAllLinks(state.currentView);
    },
  },
});

export const { setView, initTopic, initUser, refreshLinks } =
  viewStateSlice.actions;

export default viewStateSlice.reducer;
