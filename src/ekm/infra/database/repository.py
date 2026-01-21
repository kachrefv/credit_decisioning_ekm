from sqlalchemy.orm import Session
from ekm.infra.database.models import BorrowerORM, ApplicationORM, DecisionORM, UserORM
from typing import List, Optional
from datetime import datetime

class CreditRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_borrower(self, borrower_data: dict):
        if 'timestamp' in borrower_data and isinstance(borrower_data['timestamp'], (int, float)):
            borrower_data['timestamp'] = datetime.utcfromtimestamp(borrower_data['timestamp'])
        db_borrower = self.db.query(BorrowerORM).filter(BorrowerORM.id == borrower_data['id']).first()
        if db_borrower:
            for key, value in borrower_data.items():
                if key == "metadata":
                    setattr(db_borrower, "extra_metadata", value)
                else:
                    setattr(db_borrower, key, value)
        else:
            data = borrower_data.copy()
            if "metadata" in data:
                data["extra_metadata"] = data.pop("metadata")
            db_borrower = BorrowerORM(**data)
            self.db.add(db_borrower)
        self.db.commit()
        self.db.refresh(db_borrower)
        return db_borrower

    def get_borrowers(self, skip: int = 0, limit: int = 100, is_trained: Optional[bool] = None) -> List[BorrowerORM]:
        query = self.db.query(BorrowerORM)
        if is_trained is not None:
            query = query.filter(BorrowerORM.is_trained == is_trained)
        return query.offset(skip).limit(limit).all()

    def get_borrower(self, borrower_id: str) -> Optional[BorrowerORM]:
        return self.db.query(BorrowerORM).filter(BorrowerORM.id == borrower_id).first()

    def get_application(self, application_id: str) -> Optional[ApplicationORM]:
        return self.db.query(ApplicationORM).filter(ApplicationORM.id == application_id).first()

    def get_decision(self, decision_id: str) -> Optional[DecisionORM]:
        return self.db.query(DecisionORM).filter(DecisionORM.id == decision_id).first()

    def save_application(self, app_data: dict):
        if 'timestamp' in app_data and isinstance(app_data['timestamp'], (int, float)):
            app_data['timestamp'] = datetime.utcfromtimestamp(app_data['timestamp'])
        db_app = self.db.query(ApplicationORM).filter(ApplicationORM.id == app_data['id']).first()
        if db_app:
            for key, value in app_data.items():
                if key == "metadata":
                    setattr(db_app, "extra_metadata", value)
                else:
                    setattr(db_app, key, value)
        else:
            data = app_data.copy()
            if "metadata" in data:
                data["extra_metadata"] = data.pop("metadata")
            db_app = ApplicationORM(**data)
            self.db.add(db_app)
        self.db.commit()
        return db_app

    def save_decision(self, dec_data: dict):
        if 'timestamp' in dec_data and isinstance(dec_data['timestamp'], (int, float)):
            dec_data['timestamp'] = datetime.utcfromtimestamp(dec_data['timestamp'])
        db_dec = self.db.query(DecisionORM).filter(DecisionORM.id == dec_data['id']).first()
        if db_dec:
            for key, value in dec_data.items():
                if key == "metadata":
                    setattr(db_dec, "extra_metadata", value)
                else:
                    setattr(db_dec, key, value)
        else:
            data = dec_data.copy()
            if "metadata" in data:
                data["extra_metadata"] = data.pop("metadata")
            db_dec = DecisionORM(**data)
            self.db.add(db_dec)
        self.db.commit()
        return db_dec

    def get_applications(self, skip: int = 0, limit: int = 100, is_trained: Optional[bool] = None) -> List[ApplicationORM]:
        query = self.db.query(ApplicationORM)
        if is_trained is not None:
            query = query.filter(ApplicationORM.is_trained == is_trained)
        return query.offset(skip).limit(limit).all()

    def get_decisions(self, skip: int = 0, limit: int = 100, is_trained: Optional[bool] = None) -> List[DecisionORM]:
        query = self.db.query(DecisionORM)
        if is_trained is not None:
            query = query.filter(DecisionORM.is_trained == is_trained)
        return query.offset(skip).limit(limit).all()

    def mark_records_as_trained(self, borrower_ids: List[str], app_ids: List[str], dec_ids: List[str]):
        if borrower_ids:
            self.db.query(BorrowerORM).filter(BorrowerORM.id.in_(borrower_ids)).update({"is_trained": True}, synchronize_session=False)
        if app_ids:
            self.db.query(ApplicationORM).filter(ApplicationORM.id.in_(app_ids)).update({"is_trained": True}, synchronize_session=False)
        if dec_ids:
            self.db.query(DecisionORM).filter(DecisionORM.id.in_(dec_ids)).update({"is_trained": True}, synchronize_session=False)
        self.db.commit()

    def get_user_by_email(self, email: str) -> Optional[UserORM]:
        return self.db.query(UserORM).filter(UserORM.email == email).first()

    def get_user_by_id(self, user_id: str) -> Optional[UserORM]:
        return self.db.query(UserORM).filter(UserORM.id == user_id).first()

    def create_user(self, user_data: dict) -> UserORM:
        db_user = UserORM(**user_data)
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    def update_user(self, user_id: str, update_data: dict) -> Optional[UserORM]:
        db_user = self.get_user_by_id(user_id)
        if db_user:
            for key, value in update_data.items():
                setattr(db_user, key, value)
            self.db.commit()
            self.db.refresh(db_user)
        return db_user

