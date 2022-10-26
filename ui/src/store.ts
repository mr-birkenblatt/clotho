import { configureStore } from '@reduxjs/toolkit';
import lineStateSliceReducer from './main/LineStateSlice';

export default configureStore({
  reducer: {
    lineState: lineStateSliceReducer,
  },
})
