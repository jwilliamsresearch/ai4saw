"""Startup splash screens for research mode.

Two full-screen splash panels shown on every boot:
  Splash 1 — AI4SAW flame/word-art logo
  Splash 2 — AI4SAW block/bold logo

Both are centered with whitespace, shown for 5 seconds each.
"""

from __future__ import annotations

import time

from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


# ── Logos ──────────────────────────────────────────────────────────────────────

_LOGO_1 = """\
      .                                  -#               -%
      *.                                  %                #
      *.      *   +  #    -  *  :-  *:==  %.  *.  +=   %   #   #-  *:. =-   +  -.
      *.     *.   +-  #  .  #    %  ++    %   :#  ==   #   #   *   :+   %  %    #
      *.   - *        +:    #       ++    #   :#  ==   #   #   *   :+   %  #
      *.   #  =.  :    *.    *   :  **    %   -#   *  .#   #   #.  -+  .%   +   :


         +=                                           +:
      *      +                  -.                    %
     *:         *   *  -+.  #   #.   +-+.  +   +      %    =   +   %.:.
     *         *.   #  :*   %   #.   =*   %    *      %  .#    +*  %
     :*      = *       :*   %   #.   =*   %           %  .#    ++  %
       #    .   +   *  :*   %   --   +*    #   +      %    +  .+   %


     .....                                              :#
      *.   #                                             #
      *.   #   *   #  +   =  *   *  %   #   % == .+   %  #-  ==      *   *   #.  :=
      *: =+   *:   *= ++.   *+   -=     #.  %    %       #    %     %     #  #    #
      *.  #=  +           # +=      #   *.  %    %       #    %     %    .#  #    #
      #:   #-  -:  -  *   =  .-  :  #. .+-  %.    +  ..  #.  .%      *  .=   #.  .%


       =*:    #                                          #:             ===   -   -=-
     -        +                                                          #    *:   :
      #-      +   #  .+  %       =   =  =--   #    .     #.  #.   #      =:  + #      #  +.  :*=
         +#-  *       %   *  +  %    #  -.     #  .      #.  #.   %       *    =: :      =-  .*
     +     #  *   *   %   --    #       -.     =:.       #.  #.   %       * -   #    --  --  .*
     +-   -   *.  *  .%    #     *   =  +:      +.       #:  #:   #       .#    -#   .+  =+  -*
                                                .
                                             *#"""

_LOGO_2 = """\
                           *
     .**. :****  +*-       *
     ****+*****+- .+       *      +++.:        :. +     =               -. -           .+
     ***************       *       * .=.+ + *  :: * *.+ + = ::=.= +- +. -.-+ = .:  *.* +-
     ***************       *       * .- * =    :: * * = +  += *   =.  ::-. = = .:  + * -.
      -*****+ ...:+        *                                                   ::
      :*: **+  -.          *
      =*: **+  --          *      :*.   *.        +.  -=  =:               *
      +*. **+  =-          *      :**-  *.  ***. ***-+*** += *****   ****: ****=  ****  -********
      *******              *      :* +- *. *   *. *.  *=  +- *-  *: *.  *: *   *.  ..=* ==  *+  *:
     ********:             *      :*  =+*.:*   *. *.  *=  += *:  *:.*.  *: *   *..*  -* =+  *+  +:
     ********:             *      :*   -*. *+ -+  *=. **. +- *-  *: =*..*: *   *..*. -* ==  *+  +:
     -****+.               *                                            *.
   .       .**+-..::       *                                        :**+.
                           *"""

_FOOTER_ART = (
    "░█▀█░▀█▀░█░█░█▀▀░█▀█░█░█\n"
    "░█▀█░░█░░░▀█░▀▀█░█▀█░█▄█\n"
    "░▀░▀░▀▀▀░░░▀░▀▀▀░▀░▀░▀░▀"
)
_CREDIT = "AI4SAW by James Williams  ·  AI for Slavery and War  ·  University of Nottingham"


# ── Core display ────────────────────────────────────────────────────────────────

def _show_logo_fullscreen(logo: str, console: Console, seconds: float = 5.0) -> None:
    """Display a logo centered with whitespace, full screen, for `seconds` seconds."""
    text = Text(logo, style="bright_white", overflow="fold")
    renderable = Align.center(text, vertical="middle")
    with Live(renderable, screen=True, console=console, refresh_per_second=1):
        time.sleep(seconds)
    # Live exits → original screen restored


# ── Public entry points ─────────────────────────────────────────────────────────

def show_splash_1(console: Console) -> None:
    """Splash 1 — flame/word-art logo, 5 seconds."""
    _show_logo_fullscreen(_LOGO_1, console)


def show_splash_2(console: Console) -> None:
    """Splash 2 — block/bold logo, 5 seconds."""
    _show_logo_fullscreen(_LOGO_2, console)


def prompt_query(console: Console) -> str:
    """Show logo 1 in normal terminal then prompt for the research query."""
    console.clear()
    print(_LOGO_1)
    console.print(Rule(style="blue"))
    footer = Text(escape(_FOOTER_ART), style="cyan")
    footer.append(f"\n{_CREDIT}", style="dim italic")
    console.print(Align.center(footer))
    console.print()
    query = console.input("[bold blue]  Research query:[/bold blue] ").strip()
    if not query:
        raise SystemExit("No query provided.")
    return query


def make_processing_layout(query: str, steps: list[tuple[str, str]]) -> "Layout":
    """Full-screen Layout for the Initialising Research screen."""
    from rich.layout import Layout

    layout = Layout()
    layout.split_column(
        Layout(name="top",    ratio=1),
        Layout(name="panel",  size=max(8, len(steps) + 7)),
        Layout(name="bottom", ratio=1),
    )
    layout["top"].update(Text(""))
    layout["bottom"].update(Text(""))

    lines = Text()
    lines.append("  Query: ", style="dim")
    lines.append(f'"{query}"\n\n', style="bold white")
    for style, msg in steps:
        lines.append(f"{msg}\n", style=style)

    layout["panel"].update(
        Panel(
            lines,
            title="[bold blue]  Initialising Research  [/bold blue]",
            border_style="blue",
            padding=(1, 4),
        )
    )
    return layout
