import { Badge, Group, Paper, Text } from "@mantine/core";
import type { ReactNode } from "react";
export function Panel({
  title,
  badge,
  children,
  tools,
  className = "",
}: {
  title: string;
  badge?: string;
  children: ReactNode;
  tools?: ReactNode;
  className?: string;
}) {
  return (
    <Paper
      component="section"
      className={`panel ${className}`}
      withBorder
      radius="lg"
    >
      <Group justify="space-between" className="panel-heading">
        <Group gap="xs">
          <Text component="h2" fw={600} size="sm">
            {title}
          </Text>
          {badge && (
            <Badge variant="light" color="gray" size="sm">
              {badge}
            </Badge>
          )}
        </Group>
        {tools}
      </Group>
      {children}
    </Paper>
  );
}
