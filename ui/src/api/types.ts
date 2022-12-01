import { MHash, ValidLink, Votes } from '../graph/keys';

export type Token = string & { _token: void };

export type UserId = string & { _userId: void };
export type Username = string & { _username: void };

type UserPermissions = {
  canCreateTopic: Readonly<boolean>;
};
export type User = {
  token: Readonly<Token>;
  name: Readonly<Username>;
  userId: Readonly<UserId>;
  permissions: Readonly<UserPermissions>;
};

type ApiUserPermissions = {
  can_create_topic: Readonly<boolean>;
};
export type ApiLoginResponse = {
  token: Readonly<Token>;
  user: Readonly<Username>;
  userid: Readonly<UserId>;
  permissions: Readonly<ApiUserPermissions>;
};

export function toUser(userResp: Readonly<ApiLoginResponse>): Readonly<User> {
  const { user, userid, permissions, ...rest } = userResp;
  return {
    name: user,
    userId: userid,
    permissions: {
      canCreateTopic: permissions.can_create_topic,
    },
    ...rest,
  };
}

export type ApiTopic = {
  topics: Readonly<{ [key: string]: string }>;
  next: Readonly<number>;
};

export type ApiRead = {
  messages: Readonly<{ [key: string]: string }>;
  skipped: Readonly<MHash[]>;
};

export type ApiLinkResponse = {
  parent: Readonly<MHash>;
  child: Readonly<MHash>;
  user: Readonly<Username> | undefined;
  userid: Readonly<UserId> | undefined;
  first: Readonly<number>;
  votes: Votes;
};

export function toLink(link: Readonly<ApiLinkResponse>): Readonly<ValidLink> {
  const { user, userid, ...rest } = link;
  return {
    username: user,
    userId: userid,
    ...rest,
  };
}

export type ApiLinkList = {
  links: Readonly<Readonly<ApiLinkResponse>[]>;
  next: Readonly<number>;
};

export function toLinks(
  links: Readonly<Readonly<ApiLinkResponse>[]>,
): Readonly<Readonly<ValidLink>[]> {
  return links.map((link) => toLink(link));
}
