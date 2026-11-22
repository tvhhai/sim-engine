import { Children, isValidElement, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  Alert,
  Anchor,
  Blockquote,
  Button,
  Loader,
  ScrollArea,
  Select,
  Table,
  Text,
  Typography,
  useComputedColorScheme,
} from "@mantine/core";
import { FlaskConical, MonitorPlay } from "lucide-react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { useStudio } from "../studio/StudioContext";
import { useRoute } from "../studio/router";
import {
  docAssetUrl,
  episodeIndex,
  headings,
  parseStudioLink,
  slugify,
} from "../studio/docs";
import { MermaidDiagram } from "../components/MermaidDiagram";

type DocInfo = { name: string; title: string };

function textOf(node: ReactNode): string {
  return Children.toArray(node)
    .map((child) =>
      typeof child === "string" || typeof child === "number"
        ? String(child)
        : isValidElement<{ children?: ReactNode }>(child)
          ? textOf(child.props.children)
          : "",
    )
    .join("");
}

export default function DocsScreen() {
  const { catalog, loadCase, setError } = useStudio();
  const route = useRoute();
  const dark = useComputedColorScheme("dark") === "dark";
  const [docs, setDocs] = useState<DocInfo[] | null>(null),
    [markdown, setMarkdown] = useState(""),
    [failure, setFailure] = useState("");
  const name = route.params.get("doc") || docs?.[0]?.name || "";

  useEffect(() => {
    fetch("/studio/docs")
      .then((r) => r.json())
      .then((d: { docs: DocInfo[] }) => setDocs(d.docs))
      .catch((e: Error) => setFailure(e.message));
  }, []);
  useEffect(() => {
    if (!name) return;
    setMarkdown("");
    fetch(`/studio/docs/${encodeURIComponent(name)}.md`)
      .then((r) => {
        if (!r.ok) throw new Error(`Không có hướng dẫn "${name}"`);
        return r.text();
      })
      .then(setMarkdown)
      .catch((e: Error) => setFailure(e.message));
  }, [name]);
  // Mở link có #mục: cuộn tới đúng tiêu đề sau khi nội dung đã dựng xong
  const section = route.params.get("section");
  useEffect(() => {
    if (markdown && section)
      document.getElementById(section)?.scrollIntoView({ block: "start" });
  }, [markdown, section]);

  const toc = useMemo(() => headings(markdown), [markdown]);

  async function openStudioLink(href: string) {
    const link = parseStudioLink(href);
    if (!link) return;
    const index = episodeIndex(catalog, link.source, link.seed);
    if (index === null) {
      setError(
        `Seed ${link.seed} chưa có trong bộ replay. Chạy make_replay có seed này rồi mở lại.`,
      );
      return;
    }
    try {
      if (link.screen === "lab") await loadCase(link.source, index);
      else
        route.navigate("replay", {
          source: link.source,
          episode: index,
          step: link.step,
        });
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const heading =
    (Tag: "h2" | "h3") =>
    ({ children }: { children?: ReactNode }) => (
      <Tag id={slugify(textOf(children))}>{children}</Tag>
    );
  const components: Components = {
    h2: heading("h2"),
    h3: heading("h3"),
    a: ({ href, children }) => {
      const link = parseStudioLink(href);
      if (link)
        return (
          <Button
            component="span"
            role="link"
            tabIndex={0}
            size="compact-sm"
            variant="light"
            leftSection={
              link.screen === "lab" ? (
                <FlaskConical size={14} />
              ) : (
                <MonitorPlay size={14} />
              )
            }
            onClick={() => openStudioLink(href!)}
            onKeyDown={(e) => e.key === "Enter" && openStudioLink(href!)}
          >
            {children}
          </Button>
        );
      if (href?.startsWith("#"))
        return (
          <Anchor
            href={`#/docs?doc=${name}&section=${href.slice(1)}`}
            onClick={(e) => {
              e.preventDefault();
              route.navigate("docs", { doc: name, section: href.slice(1) }, true);
            }}
          >
            {children}
          </Anchor>
        );
      return (
        <Anchor href={href} target="_blank" rel="noreferrer">
          {children}
        </Anchor>
      );
    },
    img: ({ src, alt }) => (
      <img className="doc-image" src={docAssetUrl(src as string)} alt={alt} />
    ),
    table: ({ children }) => (
      <Table.ScrollContainer minWidth={480}>
        <Table withTableBorder withColumnBorders striped>
          {children}
        </Table>
      </Table.ScrollContainer>
    ),
    blockquote: ({ children }) => (
      <Blockquote color="teal" className="doc-note">
        {children}
      </Blockquote>
    ),
    code: ({ className, children }) => {
      const lang = /language-(\w+)/.exec(className || "")?.[1];
      if (lang === "mermaid")
        return <MermaidDiagram code={textOf(children).trim()} dark={dark} />;
      return <code className={className}>{children}</code>;
    },
    pre: ({ children }) => {
      // Sơ đồ mermaid không nằm trong khung <pre>
      const only = Children.toArray(children)[0];
      if (
        isValidElement<{ className?: string }>(only) &&
        /language-mermaid/.test(only.props.className || "")
      )
        return <>{children}</>;
      return <pre>{children}</pre>;
    },
  };

  return (
    <div className="standard-screen docs-screen">
      <div className="screen-toolbar">
        <div>
          <Text fw={600}>Hướng dẫn</Text>
          <Text size="sm" c="dimmed">
            Đọc từ docs/*.md (cùng file trên GitHub). Nút xanh mở thẳng Replay
            hoặc Lab.
          </Text>
        </div>
        {docs && docs.length > 1 && (
          <Select
            aria-label="Chọn hướng dẫn"
            data={docs.map((d) => ({ value: d.name, label: d.title }))}
            value={name}
            onChange={(v) => v && route.navigate("docs", { doc: v })}
            allowDeselect={false}
          />
        )}
      </div>
      {failure && (
        <Alert color="red" title="Không tải được hướng dẫn">
          {failure}
        </Alert>
      )}
      {!markdown && !failure && <Loader />}
      {markdown && (
        <div className="docs-layout">
          <nav className="docs-toc" aria-label="Mục lục">
            <ScrollArea.Autosize mah="calc(100vh - 180px)">
              {toc.map((h) => (
                <a
                  key={h.id}
                  href={`#/docs?doc=${name}&section=${h.id}`}
                  className={h.depth === 3 ? "toc-sub" : undefined}
                  onClick={(e) => {
                    e.preventDefault();
                    route.navigate("docs", { doc: name, section: h.id }, true);
                    document.getElementById(h.id)?.scrollIntoView({ block: "start" });
                  }}
                >
                  {h.text}
                </a>
              ))}
            </ScrollArea.Autosize>
          </nav>
          <Typography className="docs-body">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeRaw]}
              components={components}
              // react-markdown bỏ các scheme lạ; giữ lại studio: để mở Replay / Lab
              urlTransform={(url) =>
                url.startsWith("studio:") ? url : defaultUrlTransform(url)
              }
            >
              {markdown}
            </ReactMarkdown>
          </Typography>
        </div>
      )}
    </div>
  );
}
