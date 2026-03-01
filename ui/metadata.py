"""Pipeline metadata panel — latency, cache, guardrail flag badges."""

import streamlit as st
from ui.config import GUARDRAIL_SEVERITY


def render_metadata_panel(metadata: dict):
    """Render a collapsible panel with pipeline execution details."""
    with st.expander("Pipeline Details", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            latency = metadata.get("latency_ms", 0)
            if latency < 2000:
                st.markdown(f"**Latency:** :green[{latency:.0f}ms]")
            elif latency < 5000:
                st.markdown(f"**Latency:** :orange[{latency:.0f}ms]")
            else:
                st.markdown(f"**Latency:** :red[{latency:.0f}ms]")

        with col2:
            cache_hit = metadata.get("cache_hit", False)
            label = ":green[Yes]" if cache_hit else "No"
            st.markdown(f"**Cache Hit:** {label}")

        with col3:
            viz_count = len(metadata.get("visualizations", []))
            st.markdown(f"**Charts:** {viz_count}")

        # Guardrail flags
        flags = metadata.get("guardrail_flags", [])
        if flags:
            st.markdown("---")
            st.markdown("**Guardrail Flags:**")
            for flag in flags:
                # Handle compound flags like UNGROUNDED_NUMBER:1234.56
                flag_key = flag.split(":")[0]
                severity, label = GUARDRAIL_SEVERITY.get(flag_key, ("info", flag))

                if severity == "error":
                    st.error(f"BLOCKED: {label}", icon="\U0001f6d1")
                elif severity == "warning":
                    st.warning(label, icon="\u26a0\ufe0f")
                else:
                    st.info(label, icon="\u2139\ufe0f")
