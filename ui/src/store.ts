import { configureStore } from '@reduxjs/toolkit';
import viewStateSliceReducer from './main/ViewStateSlice';

const store = configureStore({
  reducer: {
    viewState: viewStateSliceReducer,
  },
});

export default store;

export type RootState = ReturnType<typeof store.getState>;
// FIXME for reference
// ts-unused-exports:disable-next-line
export type AppDispatch = typeof store.dispatch;
