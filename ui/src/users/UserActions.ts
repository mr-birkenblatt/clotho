import { DEFAULT_PRIVILEGE_API, PrivilegeApiProvider } from '../api/privilege';
import {
  Link,
  MHash,
  Token,
  toLink,
  toUser,
  User,
  Username,
  VoteTypeExt,
} from '../api/types';

export default class UserActions {
  private readonly api: PrivilegeApiProvider;

  constructor(api?: PrivilegeApiProvider) {
    this.api = api ?? DEFAULT_PRIVILEGE_API;
  }

  async login(username: Readonly<Username>): Promise<User> {
    const res = await this.api.login(username);
    return toUser(res);
  }

  async activeUser(
    token: Readonly<Token> | undefined,
  ): Promise<User | undefined> {
    if (token === undefined) {
      return undefined;
    }
    const res = await this.api.userInfo(token);
    return toUser(res);
  }

  async vote(
    token: Readonly<Token>,
    parent: Readonly<MHash>,
    child: Readonly<MHash>,
    votes: Readonly<VoteTypeExt[]>,
    isAdd: boolean,
  ): Promise<Link> {
    const res = await this.api.vote(token, parent, child, votes, isAdd);
    return toLink(res);
  }
} // UserActions
