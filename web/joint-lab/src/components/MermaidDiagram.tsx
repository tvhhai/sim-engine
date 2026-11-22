import { useEffect, useId, useState } from "react";
import { Code, Loader, Text } from "@mantine/core";

// Sơ đồ ```mermaid trong hướng dẫn. Thư viện mermaid nặng nên chỉ tải khi màn Hướng dẫn có sơ đồ.
export function MermaidDiagram({
  code,
  dark,
}: {
  code: string;
  dark: boolean;
}) {
  const id = "mmd-" + useId().replace(/[^a-zA-Z0-9]/g, "");
  const [svg, setSvg] = useState(""),
    [error, setError] = useState("");
  useEffect(() => {
    let alive = true;
    import("mermaid")
      .then(async ({ default: mermaid }) => {
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: "strict",
          theme: dark ? "dark" : "default",
          fontFamily: "inherit",
        });
        const { svg } = await mermaid.render(id, code);
        if (alive) setSvg(svg);
      })
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
    };
  }, [code, dark, id]);
  if (error)
    return (
      <>
        <Text c="red" size="sm">
          Không vẽ được sơ đồ: {error}
        </Text>
        <Code block>{code}</Code>
      </>
    );
  if (!svg) return <Loader size="sm" />;
  return (
    <div
      className="doc-diagram"
      role="img"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  );
}
