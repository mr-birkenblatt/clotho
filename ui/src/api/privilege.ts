import { MHash, VoteType } from '../graph/keys';
import { ApiLinkResponse, LoginResponse, Token, Username } from './types';

// FIXME use
// ts-unused-exports:disable-next-line
export type PrivilegeApiProvider = {
  login: (username: Readonly<Username>) => Promise<LoginResponse>;
  userInfo: (token: Readonly<Token>) => Promise<LoginResponse>;
  vote: (
    token: Readonly<Token>,
    parent: Readonly<MHash>,
    child: Readonly<MHash>,
    votes: Readonly<VoteType[]>,
    isadd: Readonly<boolean>,
  ) => Promise<ApiLinkResponse>;
};
