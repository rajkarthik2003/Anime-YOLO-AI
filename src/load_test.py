import asyncio
import aiohttp
import time
import statistics

# Simple load test for /predict endpoint
URL = "http://localhost:8000/predict"
TEST_IMAGE = "data/raw/images/val/sample.jpg"  # Update with actual image path
CONCURRENT = 10
TOTAL_REQUESTS = 100

async def send_request(session, img_path):
    start = time.time()
    try:
        with open(img_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename='test.jpg')
            async with session.post(URL, data=data) as resp:
                await resp.json()
                latency = time.time() - start
                return latency, resp.status
    except Exception as e:
        return time.time() - start, 0

async def run_load_test():
    latencies = []
    statuses = []
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, TEST_IMAGE) for _ in range(TOTAL_REQUESTS)]
        results = await asyncio.gather(*tasks)
        for lat, status in results:
            latencies.append(lat)
            statuses.append(status)
    
    success = sum(1 for s in statuses if s == 200)
    print(f"Total: {TOTAL_REQUESTS}, Success: {success}, Errors: {TOTAL_REQUESTS - success}")
    print(f"Latency (s): min={min(latencies):.3f}, max={max(latencies):.3f}, avg={statistics.mean(latencies):.3f}, p95={statistics.quantiles(latencies, n=20)[18]:.3f}")

if __name__ == '__main__':
    asyncio.run(run_load_test())
