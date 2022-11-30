import { configureStore } from '@reduxjs/toolkit';
import viewStateSliceReducer from './main/ViewStateSlice';
import userStateSliceReducer from './users/UserStateSlice';

const store = configureStore({
  reducer: {
    viewState: viewStateSliceReducer,
    userState: userStateSliceReducer,
  },
});

export default store;

export type RootState = ReturnType<typeof store.getState>;
// FIXME for reference
// ts-unused-exports:disable-next-line
export type AppDispatch = typeof store.dispatch;
