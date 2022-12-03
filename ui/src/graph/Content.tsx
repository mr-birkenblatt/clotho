import { PureComponent, ReactElement } from 'react';
import ReactMarkdown from 'react-markdown';
import styled from 'styled-components';
import { NormalComponents } from 'react-markdown/lib/complex-types';
import { SpecialComponents } from 'react-markdown/lib/ast-to-react';

const ContentBox = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;

  height: var(--md-size);
  width: var(--md-size);
  overflow-y: auto;
  overflow-x: hidden;
  word-break: break-all;
`;

const Link = styled.a`
  color: var(--md-anchor);

  &:visited {
    color: var(--md-anchor);
  }
  &:active {
    color: var(--md-anchor);
  }
  &:hover {
    color: var(--md-anchor-hover);
  }
`;

const MD_COMPONENTS: Partial<
  Omit<NormalComponents, keyof SpecialComponents> & SpecialComponents
> = {
  a: ({ node: _, ...props }) => <Link {...props} />,
};

type ContentProps = { children: string };

export default class Content extends PureComponent<ContentProps> {
  render(): ReactElement {
    const { children } = this.props;
    return (
      <ContentBox>
        <ReactMarkdown
          skipHtml={true}
          components={MD_COMPONENTS}>
          {children}
        </ReactMarkdown>
      </ContentBox>
    );
  }
} // Content
