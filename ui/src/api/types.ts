import { MHash, ValidLink, Votes } from '../graph/keys';

export type Token = string & { _token: void };

export type UserId = string & { _userId: void };
export type Username = string & { _username: void };
// FIXME use
// ts-unused-exports:disable-next-line
export type UserPermissions = {
  canCreateTopic: Readonly<boolean>;
};
type ApiUserPermissions = {
  can_create_topic: Readonly<boolean>;
};
// FIXME use
// ts-unused-exports:disable-next-line
export type User = {
  userId: Readonly<UserId>;
  name: Readonly<Username>;
  token: Readonly<Token>;
  permissions: Readonly<UserPermissions>;
};

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

export type LoginResponse = {
  token: Readonly<Token>;
  user: Readonly<Username>;
  userid: Readonly<UserId>;
  permissions: Readonly<ApiUserPermissions>;
};
