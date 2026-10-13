from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    name: str
    type: str
    default: float
    min: float
    max: float


# Parameters each strategy accepts. Kept apart from the behaviours so validation does not need pandas.
STRATEGY_PARAMETERS: dict[str, tuple[Parameter, ...]] = {
    "buy-and-hold": (),
    "kronos-top5": (Parameter("top_k", "integer", 5, 1, 20),),
    "ma-crossover": (
        Parameter("fast", "integer", 10, 2, 100),
        Parameter("slow", "integer", 50, 5, 250),
        Parameter("stop_loss", "number", 0.05, 0, 0.5),
    ),
}
