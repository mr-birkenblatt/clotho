import { configureStore } from '@reduxjs/toolkit';
import lineStateSliceReducer from './lineStateSlice';

export default configureStore({
  reducer: {
    lineState: lineStateSliceReducer,
  },
})
