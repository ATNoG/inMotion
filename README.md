# inMotion

## Dataset Labels:
> A - Inside the Bus

> B - Outside the Bus

## Route Labels:
> LoopA - Inside of the bus

> LoopB - Outside of the bus

> AB - From inside to outside the bus

> BA - From outside to inside the bus

## Dataset Structure:

Each line represents a sequence of RSSI measurements captured every second over 10 seconds, associated with a MAC address and a label indicating the route taken.

---

### Dataset Example

| MAC Address         | 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   | 10  | Trajeto |
|----------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|----------|
| e6:53:5c:2a:e8:e2 | -48.0 | -48.0 | -54.0 | -53.0 | -55.0 | -50.0 | -57.0 | -53.0 | -55.0 | -51.0 | LoopB |
| e6:53:5c:2a:e8:e2 | -41.0 | -31.0 | -31.0 | -35.0 | -40.0 | -48.0 | -45.0 | -48.0 | -54.0 | -56.0 | AB |


---

## SCHEMA:
![Esquema](images/Esquema.png)