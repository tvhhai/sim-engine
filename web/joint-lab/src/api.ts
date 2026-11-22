export async function api<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(
    path,
    body === undefined
      ? { signal }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal,
        },
  );
  if (!response.ok) throw new Error(`Máy chủ trả lỗi ${response.status}`);
  const result = await response.json();
  if (result.error && !result.status) throw new Error(result.error);
  return result;
}
