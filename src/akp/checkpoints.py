from contextlib import AsyncExitStack

from langgraph.checkpoint.memory import InMemorySaver

from akp.config import Settings


async def open_checkpointer(settings: Settings, stack: AsyncExitStack):
    """Checkpointer that saves graph state after every step, so a run that fails
    midway can be resumed from the failed step instead of starting over."""
    if settings.checkpointer == "redis":
        from langgraph.checkpoint.redis.aio import AsyncRedisSaver

        saver = await stack.enter_async_context(
            AsyncRedisSaver.from_conn_string(
                settings.redis_url,
                ttl={"default_ttl": settings.checkpoint_ttl_minutes, "refresh_on_read": True},
            )
        )
        await saver.asetup()
        return saver
    if settings.checkpointer == "memory":
        return InMemorySaver()
    return None
