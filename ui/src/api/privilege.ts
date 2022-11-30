import { MHash, VoteType } from '../graph/keys';
import { URL_PREFIX } from '../misc/constants';
import { json, toJson } from '../misc/util';
import { ApiLinkResponse, LoginResponse, Token, Username } from './types';

// FIXME use
// ts-unused-exports:disable-next-line
export type PrivilegeApiProvider = {
  login: (user: Readonly<Username>) => Promise<LoginResponse>;
  userInfo: (token: Readonly<Token>) => Promise<LoginResponse>;
  vote: (
    token: Readonly<Token>,
    parent: Readonly<MHash>,
    child: Readonly<MHash>,
    votes: Readonly<VoteType[]>,
    isadd: Readonly<boolean>,
  ) => Promise<ApiLinkResponse>;
};

// FIXME use
// ts-unused-exports:disable-next-line
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
