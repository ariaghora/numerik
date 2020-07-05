unit ndarray;

{$mode delphi}
{$H+}
{$modeSwitch advancedRecords}

interface

uses
  Classes, SysUtils, math;

type
  TLongVector = array of longint;
  TSingleVector = array of single;
  TUFunc = function(a: single): single;
  TBFunc<T> = function(a, b: T): T;

type
  TFTensor = record
  private
    FIsContiguous: boolean;
    function GetNDims: integer;
    function GetSize: longint;
    function IndexToStridedOffset(Index: array of longint): longint;
    function OffsetToStrided(Offset: longint): longint;
    function ShapeToStrides(AShape: array of longint): TLongVector;
  public
    Data:  array of single;
    Shape: TLongVector;
    Strides: TLongVector;
    function Contiguous: TFTensor;
    function Get(i: longint): single; overload;
    function Get(i: array of integer): single; overload;
    function Reshape(NewShape: array of longint): TFTensor;
    property IsContiguous: boolean read FIsContiguous write FIsContiguous;
    property NDims: integer read GetNDims;
    property Size: longint read GetSize;

    class operator Add(A, B: TFTensor): TFTensor;
    class operator Subtract(A, B: TFTensor): TFTensor;
  end;

  TBroadcastResult = record
    A, B: TFTensor;
  end;


  function BroadcastArrays(A, B: TFTensor): TBroadcastResult;
  function CreateEmptyFTensor(Contiguous: boolean = True): TFTensor;
  function CreateFTensor(AData: array of single): TFTensor;
  function AsStrided(A: TFTensor; Shape, Strides: array of longint): TFTensor;
  function ApplyBFunc(A, B: TFTensor; BFunc: TBFunc<single>): TFTensor;
  function DynArrayToVector(A: array of longint): TLongVector;
  procedure PrintTensor(A: TFTensor);

  function Prod<T>(Data: array of T): T;
  function Reverse<T>(Data: T): T;
  function VectorsEqual<T>(A, B: T): boolean;
  procedure PrintVector<T>(Data: T);

  { Arithmetic function wrappers }
  function Add(a, b: single): single;
  function Multiply(a, b: single): single;
  function Subtract(a, b: single): single;

implementation

  function BroadcastArrays(A, B: TFTensor): TBroadcastResult;
  var
    AdjShapeA, AdjShapeB, AdjStridesA, AdjStridesB: TLongVector;
    i, NewNDims: integer;
  begin
    NewNDims := Max(A.NDims, B.NDims);
    AdjShapeA := Reverse<TLongVector>(A.Shape);
    AdjShapeB := Reverse<TLongVector>(B.Shape);
    AdjStridesA := Reverse<TLongVector>(A.Strides);
    AdjStridesB := Reverse<TLongVector>(B.Strides);

    { Broadcast dimension is always max(A.NDims, B.NDims) }
    if A.NDims < NewNDims then
    begin
      SetLength(AdjShapeA, NewNDims);
      SetLength(AdjStridesA, NewNDims)
    end
    else
    begin
      SetLength(AdjShapeB, NewNDims);
      SetLength(AdjStridesB, NewNDims)
    end;

    for i := 0 to NewNDims - 1 do
      if (AdjShapeA[i] <> AdjShapeB[i]) then
      begin
        if (AdjShapeA[i] < AdjShapeB[i]) and (AdjShapeA[i] <= 1) then
        begin
          AdjStridesA[i] := 0;
          AdjShapeA[i] := AdjShapeB[i]
        end
        else if (AdjShapeB[i] < AdjShapeA[i]) and (AdjShapeB[i] <= 1) then
        begin
          AdjStridesB[i] := 0;
          AdjShapeB[i] := AdjShapeA[i]
        end
        else
          raise Exception.Create('Broadcasting failed.');
      end;

    Result.A := AsStrided(A, Reverse<TLongVector>(AdjShapeA),
                             Reverse<TLongVector>(AdjStridesA));
    Result.B := AsStrided(B, Reverse<TLongVector>(AdjShapeB),
                             Reverse<TLongVector>(AdjStridesB));
  end;

  function CreateEmptyFTensor(Contiguous: boolean = True):TFTensor;
  begin
    Result.IsContiguous := Contiguous;
  end;

  function CreateFTensor(AData: array of single): TFTensor;
  var
    i: longint;
  begin
    Result := CreateEmptyFTensor(True);
    SetLength(Result.Data, Length(AData));
    for i := 0 to Length(AData) - 1 do
      Result.Data[i] := AData[i];
    SetLength(Result.Shape, 1);
    Result.Shape[0] := Length(AData);
    Result.Strides := Result.ShapeToStrides(Result.Shape);
  end;

  function AsStrided(A: TFTensor; Shape, Strides: array of longint): TFTensor;
  begin
    Result := CreateEmptyFTensor(false);
    Result.Data := A.Data;
    Result.Shape := DynArrayToVector(Shape);
    Result.Strides := DynArrayToVector(Strides);
  end;

  function ApplyBFunc(A, B: TFTensor; BFunc: TBFunc<single>): TFTensor;
  var
    i: longint;
    BcastResult: TBroadcastResult;
  begin
    if not(VectorsEqual<TLongVector>(A.Shape, B.Shape)) then
    begin
      BcastResult := BroadcastArrays(A, B);
      A := BcastResult.A;
      B := BcastResult.B;
    end;

    Result := CreateEmptyFTensor(True);
    SetLength(Result.Data, A.Size);
    Result.Shape := A.Shape;
    Result := Result.Reshape(A.Shape); // should be reset strides
    for i := 0 to A.Size - 1 do
      Result.Data[i] := BFunc(A.Get(i), B.Get(i));
  end;

  function DynArrayToVector(A: array of longint): TLongVector;
  var
    i: longint;
  begin
    Result := nil;
    SetLength(Result, Length(A));
    for i := 0 to High(A) do
      Result[i] := A[i];
  end;

  procedure PrintTensor(A: TFTensor);
  var
    r, c, i: longint;
  begin
    if A.NDims = 1 then
    begin
      for i := 0 to A.Size - 1 do
        Write(A.Get(i), ' ');
      WriteLn;
    end;

    if A.NDims = 2 then
    begin
      for r := 0 to A.Shape[0] - 1 do
      begin
        for c := 0 to A.Shape[1] - 1 do
        begin
          Write(A.Get([r, c]));
          if c < A.Shape[1] - 1 then Write(' ');
        end;
        WriteLn;
      end;
    end;
  end;

  function Prod<T>(Data: array of T): T;
  var
    x: T;
  begin
    Result := 1;
    for x in Data do
      Result := Result * x;
  end;

  function Reverse<T>(Data: T): T;
  var
    i: longint;
  begin
    Result := nil;
    SetLength(Result, Length(Data));
    For i := High(Data) downto 0 do
      Result[High(Data) - i] := Data[i];
  end;

  function VectorsEqual<T>(A, B: T): boolean;
  var
    i: longint;
  begin
    if (Length(A) <> Length(B)) then Exit(false);
    Result := True;
    for i := 0 to High(A) do
      Result := Result and (A[i] = B[i]);
  end;

  procedure PrintVector<T>(Data: T);
  var
    i: longint;
  begin
    for i := 0 to Length(Data) - 1 do
    begin
      Write(Data[i]);
      if i < Length(Data) - 1 then
        Write(', ');
    end;
    WriteLn;
  end;

  function TFTensor.ShapeToStrides(AShape: array of longint): TLongVector;
  var
    k, j, prod: integer;
  begin
    Result := nil;

    SetLength(Result, Length(AShape));
    for k := 0 to Length(Result) - 1 do
    begin
      prod := 1;
      for j := k + 1 to Length(Result) - 1 do
        prod := prod * AShape[j];
      Result[k] := prod;
    end;
  end;

  function TFTensor.OffsetToStrided(Offset: longint): longint;
  var
    r, c: longint;
  begin
    if NDims > 2 then
      raise ENotImplemented.Create('Item access with not implemented yet with NDims > 2.');

    if NDims = 1 then Exit(Offset);
    if NDims = 2 then
    begin
      r := Offset div Shape[1];
      c := Offset mod Shape[1];
      Exit(IndexToStridedOffset([r, c]));
    end;
  end;

  function TFTensor.GetNDims: integer;
  begin
    Exit(Length(Shape));
  end;

  function TFTensor.GetSize: longint;
  begin
    Exit(Prod<longint>(Shape));
  end;

  function TFTensor.IndexToStridedOffset(Index: array of longint): longint;
  var
    i: integer;
  begin
    Result := 0;
    for i := 0 to Length(Index) - 1 do
      Result := Result + (Strides[i] * Index[i]);
  end;

  function TFTensor.Contiguous: TFTensor;
  begin
    if IsContiguous then Exit(self);
  end;

  function TFTensor.Get(i: longint): single; overload;
  begin
    if IsContiguous then Exit(Data[i]);
    Exit(Data[OffsetToStrided(i)]);
  end;

  function TFTensor.Get(i: array of integer): single; overload;
  begin
    Assert(length(i) = length(Shape), 'Index dimension does not match array dimension.');
    Exit(Data[IndexToStridedOffset(i)]);
  end;

  function TFTensor.Reshape(NewShape: array of longint): TFTensor;
  var
    i: integer;
  begin
    Assert((Prod<longint>(NewShape)) = (Prod<longint>(Shape)), 'Impossible reshape.');
    SetLength(Shape, Length(NewShape));
    for i := 0 to Length(NewShape) - 1 do
      Shape[i] := NewShape[i];
    Strides := ShapeToStrides(Shape);
    Exit(self);
  end;

  class operator TFTensor.Add(A, B: TFTensor): TFTensor;
  begin
    Exit(ApplyBFunc(A, B, @Add))
  end;

  class operator TFTensor.Subtract(A, B: TFTensor): TFTensor;
  begin
    Result := ApplyBFunc(A, B, @Subtract)
  end;

  function Add(a, b: single): single;
  begin
    Exit(a + b);
  end;

  function Multiply(a, b: single): single;
  begin
    Exit(a * b);
  end;

  function Subtract(a, b: single): single;
  begin
    Exit(a - b);
  end;

end.

