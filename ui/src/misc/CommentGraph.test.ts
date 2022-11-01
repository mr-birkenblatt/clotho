import { ApiProvider } from "./CommentGraph";

const TOPICS = {hta: "t/a", htb: "t/b"};
const MESSAGES = {
  ...TOPICS,
};

const TEST_API: ApiProvider = {
  topic: async () => ({topics: TOPICS}),
  read: async (hashes) => {
    const ms = Array.from(hashes);
    // TODO
    return {
      messages: [],
      skipped: [],
    };
  },
  link: async (linkKey, offset, limit) => {
    const { mhash, isGetParent } = linkKey;
    const query = isGetParent ? { child: mhash } : { parent: mhash };
    const url = `${URL_PREFIX}/${isGetParent ? 'parents' : 'children'}`;
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
  };

test('comment graph tests', () => {

});
