const { MongoClient } = require("mongodb");
const config = require("./config");

let client;
let db;

async function getDb() {
  if (db) return db;
  if (!config.mongoUri) {
    throw new Error("MONGO_URI is not configured");
  }

  client = new MongoClient(config.mongoUri);
  await client.connect();
  db = client.db(config.mongoDbName);
  return db;
}

async function closeDb() {
  if (client) {
    await client.close();
    client = null;
    db = null;
  }
}

module.exports = { getDb, closeDb };
