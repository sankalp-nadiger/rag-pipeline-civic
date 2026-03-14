FROM node:20

RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app
COPY . .

RUN cd backend && npm install
RUN pip install -r backend/requirements.txt

WORKDIR /app/backend

CMD ["npm", "start"]