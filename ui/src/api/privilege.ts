import { URL_PREFIX } from '../misc/constants';
import { json, toJson } from '../misc/util';
import {
  ApiLinkResponse,
  ApiLoginResponse,
  MHash,
  Token,
  Username,
  VoteTypeExt,
} from './types';

export type PrivilegeApiProvider = {
  login: (user: Readonly<Username>) => Promise<ApiLoginResponse>;
  userInfo: (token: Readonly<Token>) => Promise<ApiLoginResponse>;
  vote: (
    token: Readonly<Token>,
    parent: Readonly<MHash>,
    child: Readonly<MHash>,
    votes: Readonly<VoteTypeExt[]>,
    isadd: Readonly<boolean>,
  ) => Promise<ApiLinkResponse>;
};

export const DEFAULT_PRIVILEGE_API: PrivilegeApiProvider = {
  login: async (user) => {
    return fetch(`${URL_PREFIX}/login`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: toJson({ user }),
    }).then(json);
  },
  userInfo: async (token) => {
    return fetch(`${URL_PREFIX}/user`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: toJson({ token }),
    }).then(json);
  },
  vote: async (token, parent, child, votes, isadd) => {
    return fetch(`${URL_PREFIX}/vote`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: toJson({ token, parent, child, votes, isadd }),
    }).then(json);
  },
};
