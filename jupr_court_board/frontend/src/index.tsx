import React, { useEffect } from "react";
import ReactDOM from "react-dom/client";
import { Streamlit, withStreamlitConnection, ComponentProps } from "streamlit-component-lib";
import CourtBoard, { Court } from "./CourtBoard";
import "./styles.css";

type Props = ComponentProps & {
  args: {
    courts: Court[];
  };
};

function App({ args }: Props) {
  const courts = Array.isArray(args?.courts) ? args.courts : [];

  const onChange = (nextCourts: Court[]) => {
    Streamlit.setComponentValue({ courts: nextCourts });
  };

  useEffect(() => {
    Streamlit.setFrameHeight(720);
  }, []);

  return (
    <div className="cb-app">
      <div className="cb-topbar">
        <div className="cb-title">Court Board</div>
        <div className="cb-subtitle">Drag to reorder. Drop onto a player to bench them.</div>
      </div>

      <CourtBoard courts={courts} onChange={onChange} />
    </div>
  );
}

const ConnectedApp = withStreamlitConnection(App);

ReactDOM.createRoot(document.getElementById("root")!).render(<ConnectedApp />);
