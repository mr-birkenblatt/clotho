import { URL_PREFIX } from '../misc/constants';
import { IsGet, LinkKey, MHash, UserKey } from '../graph/keys';
import { json, toJson } from '../misc/util';
import {
  ApiLinkList,
  ApiLinkResponse,
  ApiRead,
  ApiTopic,
  Token,
} from './types';

export type GraphApiProvider = {
  topic: (offset: number, limit: number) => Promise<ApiTopic>;
  read: (hashes: Set<Readonly<MHash>>) => Promise<ApiRead>;
  link: (
    linkKey: Readonly<LinkKey>,
    offset: number,
    limit: number,
  ) => Promise<ApiLinkList>;
  userLink: (
    userKey: Readonly<UserKey>,
    offset: number,
    limit: number,
  ) => Promise<ApiLinkList>;
  singleLink: (
    parent: Readonly<MHash>,
    child: Readonly<MHash>,
    token: Readonly<Token> | undefined,
  ) => Promise<ApiLinkResponse>;
};

/* istanbul ignore next */
export const DEFAULT_API: GraphApiProvider = {
  topic: async (offset, limit) => {
    const query = new URLSearchParams({
      offset: `${offset}`,
      limit: `${limit}`,
    });
    return fetch(`${URL_PREFIX}/topic?${query}`, { method: 'GET' }).then(json);
  },
  read: async (hashes) => {
    return fetch(`${URL_PREFIX}/read`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: toJson({ hashes }),
    }).then(json);
  },
  link: async (linkKey, offset, limit) => {
    const { mhash, isGet } = linkKey;
    const query =
      isGet === IsGet.parent ? { child: mhash } : { parent: mhash };
    const url = `${URL_PREFIX}/${
      isGet === IsGet.parent ? 'parents' : 'children'
    }`;
    return fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ...query,
        offset,
        limit,
        scorer: 'best',
      }),
    }).then(json);
  },
  userLink: async (userChildKey, offset, limit) => {
    const { userId } = userChildKey;
    const url = `${URL_PREFIX}/userlinks`;
    return fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user: userId,
        offset,
        limit,
        scorer: 'best',
      }),
    }).then(json);
  },
  singleLink: async (parent, child, token) => {
    const url = `${URL_PREFIX}/link`;
    return fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        parent,
        child,
        token,
      }),
    }).then(json);
  },
};
