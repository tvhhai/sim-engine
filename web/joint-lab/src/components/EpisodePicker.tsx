import { useState } from "react";
import { Combobox, Group, InputBase, Text, useCombobox } from "@mantine/core";
import type { EpisodeSummary } from "../types";
import { episodeHaystack, matchesSearch } from "../studio/replayFilters";

type Item = EpisodeSummary & { matches: boolean };

// Ô chọn episode / seed có ô tìm kiếm NẰM TRONG danh sách thả xuống (gõ số seed, tên hoặc nhãn: 1011, hong, timeout...)
export function EpisodePicker({
  label,
  items,
  value,
  onChange,
  width = 240,
}: {
  label: string;
  items: Item[];
  value: number;
  onChange: (index: number) => void;
  width?: number;
}) {
  const [search, setSearch] = useState("");
  const combobox = useCombobox({
    onDropdownClose: () => {
      combobox.resetSelectedOption();
      setSearch("");
    },
    onDropdownOpen: () => combobox.focusSearchInput(),
  });
  const current = items.find((item) => item.index === value);
  const shown = items.filter(
    (item) => !search.trim() || matchesSearch(episodeHaystack(item), search),
  );
  const pick = (index: number) => {
    onChange(index);
    combobox.closeDropdown();
  };
  return (
    <Combobox
      store={combobox}
      withinPortal
      width={Math.max(width, 320)}
      onOptionSubmit={(v) => pick(Number(v))}
    >
      <Combobox.Target>
        <InputBase
          component="button"
          type="button"
          pointer
          label={label}
          aria-label="Chọn episode / seed"
          w={width}
          rightSection={<Combobox.Chevron />}
          rightSectionPointerEvents="none"
          onClick={() => combobox.toggleDropdown()}
        >
          <Text size="sm" truncate>
            {current?.name ?? "Chọn seed"}
          </Text>
        </InputBase>
      </Combobox.Target>
      <Combobox.Dropdown>
        <Combobox.Search
          value={search}
          onChange={(e) => setSearch(e.currentTarget.value)}
          placeholder="Tìm seed, tên, nhãn…"
          aria-label="Tìm seed"
          onKeyDown={(e) => {
            if (e.key === "Enter" && shown.length && !combobox.getSelectedOptionIndex())
              pick(shown[0].index);
          }}
        />
        <Combobox.Options mah={320} style={{ overflowY: "auto" }}>
          {shown.length ? (
            shown.map((item) => (
              <Combobox.Option
                key={item.index}
                value={String(item.index)}
                active={item.index === value}
              >
                <Group justify="space-between" wrap="nowrap" gap="xs">
                  <Text size="sm" truncate>
                    {item.name}
                  </Text>
                  <Text size="xs" c={item.success ? "teal" : "red"}>
                    {item.success ? "OK" : "FAIL"}
                    {item.matches ? "" : " · ngoài bộ lọc"}
                  </Text>
                </Group>
              </Combobox.Option>
            ))
          ) : (
            <Combobox.Empty>Không có episode khớp</Combobox.Empty>
          )}
        </Combobox.Options>
      </Combobox.Dropdown>
    </Combobox>
  );
}
