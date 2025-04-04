# Active Context

## Current Focus

*   Add functionality to specify an AWS profile when using the Bedrock LLM provider.

## Recent Changes

*   Initialized Memory Bank core files (`projectbrief.md`, `productContext.md`, `activeContext.md`, `systemPatterns.md`, `techContext.md`, `progress.md`).
*   Created `.clinerules`.

## Recent Changes

*   Initialized Memory Bank core files (`projectbrief.md`, `productContext.md`, `activeContext.md`, `systemPatterns.md`, `techContext.md`, `progress.md`).
*   Created `.clinerules`.
*   Added `--aws-profile` CLI option (and `AWS_PROFILE` env var support) to `sdeul/cli.py` for specifying AWS credentials profile for Bedrock.
*   Confirmed `sdeul/llm.py` already supported passing the profile name to `ChatBedrockConverse`.

## Next Steps

1.  Update `techContext.md` and `progress.md` in the Memory Bank.
2.  Update `.clinerules` if applicable (no new rules identified yet).
3.  Consider adding tests for the `--aws-profile` functionality.
4.  Complete the task.

## Active Decisions & Considerations

*   The initial Memory Bank content is based on a high-level understanding inferred from the project's file structure and stated goals. It will need refinement as more context is gathered.
