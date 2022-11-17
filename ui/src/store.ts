import { configureStore } from '@reduxjs/toolkit';
import lineStateSliceReducer from './main/LineStateSlice';
import viewStateSliceReducer from './main/ViewStateSlice';

const store = configureStore({
  reducer: {
    lineState: lineStateSliceReducer,
    viewState: viewStateSliceReducer,
  },
});

export default store;

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
