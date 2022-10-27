import { configureStore } from '@reduxjs/toolkit';
import lineStateSliceReducer from './main/LineStateSlice';

const store = configureStore({
  reducer: {
    lineState: lineStateSliceReducer,
  },
});

export default store;

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
