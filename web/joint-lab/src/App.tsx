import { StudioProvider } from "./studio/StudioContext";
import StudioShell from "./studio/StudioShell";
export default function App() {
  return (
    <StudioProvider>
      <StudioShell />
    </StudioProvider>
  );
}
